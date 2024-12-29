import os
import asyncio
from datetime import datetime, timedelta
from collections import deque
import logging

logger = logging.getLogger(__name__)

class RateLimiter:
    def __init__(self, name: str = "default"):
        self.name = name
        self.MAX_REQUESTS_PER_MINUTE = int(os.getenv(f"{name.upper()}_MAX_REQUESTS_PER_MINUTE", 0))
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv(f"{name.upper()}_MAX_CONCURRENT_REQUESTS", 0))
        self.MAX_RETRIES = 3
        self._request_timestamps = deque()
        self._request_queue = asyncio.Queue()
        self._request_semaphore = asyncio.Semaphore(
            self.MAX_CONCURRENT_REQUESTS if self.MAX_CONCURRENT_REQUESTS > 0 else 1000000
        )
        self._rate_limit_lock = asyncio.Lock()
        self.total_requests = 0
        self.rate_limited_requests = 0
        self._worker_task = None
        self._processing = True

    async def _worker(self):
        """Background worker to process queued requests"""
        while self._processing:
            try:
                future = await self._request_queue.get()
                try:
                    # Process the request with the semaphore
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

    async def _process_request(self):
        """Process a single request with rate limiting"""
        retry_count = 0
        last_exception = None
        
        while retry_count < self.MAX_RETRIES:
            try:
                await self._check_rate_limit()
                return True
            except Exception as e:
                retry_count += 1
                last_exception = e
                wait_time = (2 ** retry_count) * 1.5  # Exponential backoff
                logger.warning(
                    f"Rate limit reached for {self.name}. "
                    f"Attempt {retry_count}/{self.MAX_RETRIES}. "
                    f"Waiting {wait_time:.2f} seconds"
                )
                await asyncio.sleep(wait_time)
        
        raise Exception(f"Failed to acquire rate limit after {self.MAX_RETRIES} attempts: {last_exception}")

    async def acquire(self):
        """Queue request and wait for processing"""
        if self._worker_task is None or self._worker_task.done():
            self._processing = True
            self._worker_task = asyncio.create_task(self._worker())

        future = asyncio.Future()
        await self._request_queue.put(future)
        
        try:
            return await future
        except Exception as e:
            raise e

    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        if self.MAX_REQUESTS_PER_MINUTE == 0:
            return

        async with self._rate_limit_lock:
            now = datetime.now()
            # Remove timestamps older than 1 minute
            while self._request_timestamps and self._request_timestamps[0] < now - timedelta(minutes=1):
                self._request_timestamps.popleft()

            # Check if we're at the rate limit
            if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
                self.rate_limited_requests += 1
                wait_time = (self._request_timestamps[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    logger.warning(
                        f"Rate limit reached for {self.name}. "
                        f"Waiting {wait_time:.2f} seconds. "
                        f"Total requests: {self.total_requests}, "
                        f"Rate limited: {self.rate_limited_requests}"
                    )
                    await asyncio.sleep(wait_time)

            # Add current timestamp
            self._request_timestamps.append(now)
            self.total_requests += 1
            logger.debug(
                f"Request timestamp added for {self.name}. "
                f"Current requests in window: {len(self._request_timestamps)}"
            )

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
            "queue_size": self._request_queue.qsize()
        }