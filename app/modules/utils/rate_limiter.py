import os
import asyncio
from datetime import datetime, timedelta
from collections import deque

class RateLimiter:
    def __init__(self, name: str = "default"):
        # name parameter helps identify different rate limiters in logs
        self.name = name
        
        # Get limits from environment variables with specific names based on the limiter name
        self.MAX_REQUESTS_PER_MINUTE = int(os.getenv(f"{name.upper()}_MAX_REQUESTS_PER_MINUTE", 0))
        self.MAX_CONCURRENT_REQUESTS = int(os.getenv(f"{name.upper()}_MAX_CONCURRENT_REQUESTS", 0))
        
        self._request_timestamps = deque()
        semaphore_value = self.MAX_CONCURRENT_REQUESTS if self.MAX_CONCURRENT_REQUESTS > 0 else 1000000
        self._request_semaphore = asyncio.Semaphore(semaphore_value)
        self._rate_limit_lock = asyncio.Lock()

    async def acquire(self):
        """Acquire both rate limit and semaphore"""
        async with self._request_semaphore:
            await self._check_rate_limit()

    async def _check_rate_limit(self):
        """Check and enforce rate limits"""
        if self.MAX_REQUESTS_PER_MINUTE == 0:
            return  # No rate limiting if MAX_REQUESTS_PER_MINUTE is 0

        async with self._rate_limit_lock:
            now = datetime.now()
            # Remove timestamps older than 1 minute
            while self._request_timestamps and self._request_timestamps[0] < now - timedelta(minutes=1):
                self._request_timestamps.popleft()

            # Check if we're at the rate limit
            if len(self._request_timestamps) >= self.MAX_REQUESTS_PER_MINUTE:
                wait_time = (self._request_timestamps[0] + timedelta(minutes=1) - now).total_seconds()
                if wait_time > 0:
                    logging.warning(f"Rate limit reached for {self.name}. Waiting {wait_time:.2f} seconds")
                    await asyncio.sleep(wait_time)

            # Add current timestamp
            self._request_timestamps.append(now)