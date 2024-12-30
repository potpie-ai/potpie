import os
import asyncio
import random
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, name: str = "default"):
        self.name = name
        # Get configurable limits from environment variables with defaults
        self.MAX_REQUESTS_PER_MINUTE = int(os.getenv(f"{name.upper()}_MAX_REQUESTS_PER_MINUTE", 50))
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv(f"{name.upper()}_MAX_CONCURRENT_REQUESTS", 5))
        self.QUOTA_BACKOFF_SECONDS = int(os.getenv(f"{name.upper()}_QUOTA_BACKOFF_SECONDS", 60))
        self.MAX_QUEUE_SIZE = int(os.getenv(f"{name.upper()}_MAX_QUEUE_SIZE", 1000))
        self.MAX_FAILURES = int(os.getenv(f"{name.upper()}_MAX_FAILURES", 5))
        self.CIRCUIT_RESET_TIME = int(os.getenv(f"{name.upper()}_CIRCUIT_RESET_TIME", 300))
        
        self._request_timestamps = deque()
        self._request_queue = asyncio.Queue(maxsize=self.MAX_QUEUE_SIZE)
        self._request_semaphore = asyncio.Semaphore(
            self.MAX_CONCURRENT_REQUESTS if self.MAX_CONCURRENT_REQUESTS > 0 else 1000000
        )
        self._rate_limit_lock = asyncio.Lock()
        self.total_requests = 0
        self.rate_limited_requests = 0
        self._worker_task = None
        self._processing = True
        self._last_quota_exceeded = None
        self._consecutive_failures = 0
        self._circuit_open = False
        self._processing_lock = asyncio.Lock()

        logger.info(
            f"Initialized rate limiter '{name}' with {self.MAX_REQUESTS_PER_MINUTE} "
            f"requests/minute, {self.MAX_CONCURRENT_REQUESTS} concurrent requests, "
            f"and {self.QUOTA_BACKOFF_SECONDS}s quota backoff"
        )

    async def _worker(self):
        while self._processing:
            try:
                future = await self._request_queue.get()
                try:
                    async with self._request_semaphore:
                        result = await self._process_request()
                        future.set_result(result)
                except Exception as e:
                    future.set_exception(e)
                finally:
                    self._request_queue.task_done()
            except asyncio.CancelledError:
                break
            except Exception as e:
                logger.error(f"Error in worker: {e}")
                continue

    async def acquire(self):
        """Queue request and wait for processing"""
        async with self._processing_lock:  # Ensure sequential processing
            if self._worker_task is None or self._worker_task.done():
                self._processing = True
                self._worker_task = asyncio.create_task(self._worker())

            future = asyncio.Future()
            await self._request_queue.put(future)
            
            try:
                return await asyncio.wait_for(future, timeout=30)
            except asyncio.TimeoutError:
                logger.error(f"Timeout waiting for rate limiter {self.name}")
                raise Exception("Service is currently overloaded. Please try again later.")

    async def _process_request(self):
        """Process a single request with rate limiting"""
        if self._circuit_open:
            if (datetime.now() - self._last_quota_exceeded).total_seconds() < self.CIRCUIT_RESET_TIME:
                raise Exception(f"Circuit breaker open for {self.name}")
            self._circuit_open = False
            self._consecutive_failures = 0

        async with self._rate_limit_lock:
            now = datetime.now()
            # Remove timestamps older than 1 minute
            while self._request_timestamps and self._request_timestamps[0] < now - timedelta(minutes=1):
                self._request_timestamps.popleft()

            if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
                wait_time = (self._request_timestamps[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    logger.warning(f"Rate limit reached. Waiting {wait_time:.2f}s")
                    await asyncio.sleep(wait_time)

            self._request_timestamps.append(now)
            self.total_requests += 1
            return True

    async def _check_rate_limit(self):
        """Check and enforce rate limits with improved tracking"""
        if self.MAX_REQUESTS_PER_MINUTE == 0:
            return

        async with self._rate_limit_lock:
            now = datetime.now()
            # Remove timestamps older than 1 minute
            while self._request_timestamps and self._request_timestamps[0] < now - timedelta(minutes=1):
                self._request_timestamps.popleft()

            # Calculate current rate and capacity
            current_rate = len(self._request_timestamps)
            capacity_used = current_rate / self.MAX_REQUESTS_PER_MINUTE

            if current_rate >= self.MAX_REQUESTS_PER_MINUTE:
                self.rate_limited_requests += 1
                wait_time = (self._request_timestamps[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit reached for {self.name}. "
                        f"Waiting {wait_time:.2f}s. "
                        f"Capacity: {capacity_used*100:.1f}%, "
                        f"Total: {self.total_requests}, "
                        f"Limited: {self.rate_limited_requests}"
                    )
                    await asyncio.sleep(wait_time)

            # Add current timestamp
            self._request_timestamps.append(now)
            self.total_requests += 1

    def handle_quota_exceeded(self):
        """Handle quota exceeded with exponential backoff"""
        now = datetime.now()
        if self._last_quota_exceeded:
            # Calculate exponential backoff with base of 2
            time_since_last = (now - self._last_quota_exceeded).total_seconds()
            self.QUOTA_BACKOFF_SECONDS = min(
                300,  # Max backoff of 5 minutes
                self.QUOTA_BACKOFF_SECONDS * 2
            )
        self._last_quota_exceeded = now
        
        # Add jitter (±20% randomization)
        jitter = random.uniform(0.8, 1.2)
        effective_backoff = self.QUOTA_BACKOFF_SECONDS * jitter
        
        logger.warning(
            f"Quota exceeded for {self.name}. "
            f"Entering {effective_backoff:.1f}s backoff period "
            f"(base: {self.QUOTA_BACKOFF_SECONDS}s)"
        )
        return effective_backoff

    async def shutdown(self):
        """Gracefully shutdown the worker"""
        self._processing = False
        if self._worker_task:
            self._worker_task.cancel()
            try:
                await self._worker_task
            except asyncio.CancelledError:
                pass

    def get_metrics(self):
        """Return current metrics"""
        return {
            "name": self.name,
            "total_requests": self.total_requests,
            "rate_limited_requests": self.rate_limited_requests,
            "current_window_requests": len(self._request_timestamps),
            "max_requests_per_minute": self.MAX_REQUESTS_PER_MINUTE,
            "max_concurrent_requests": self.MAX_CONCURRENT_REQUESTS,
            "queue_size": self._request_queue.qsize(),
            "in_backoff": bool(self._last_quota_exceeded),
            "circuit_open": self._circuit_open,
            "consecutive_failures": self._consecutive_failures
        }