"""
Timing collector for parsing operations.
Aggregates timing data and generates summary reports.
"""
import logging
import time
from collections import defaultdict
from contextlib import contextmanager
from typing import Dict, List, Optional

logger = logging.getLogger(__name__)


class TimingCollector:
    """Collects and aggregates timing information for parsing operations."""

    def __init__(self):
        self.timings: Dict[str, List[float]] = defaultdict(list)
        self.counts: Dict[str, int] = defaultdict(int)
        self.current_stack: List[str] = []
        self.start_times: Dict[str, float] = {}

    def start(self, operation: str):
        """Start timing an operation."""
        full_operation = " > ".join(self.current_stack + [operation])
        self.current_stack.append(operation)
        self.start_times[full_operation] = time.perf_counter()

    def end(self, operation: str):
        """End timing an operation."""
        if not self.current_stack or self.current_stack[-1] != operation:
            logger.warning(f"Timing mismatch: trying to end '{operation}' but stack is {self.current_stack}")
            return

        full_operation = " > ".join(self.current_stack)
        if full_operation in self.start_times:
            elapsed = time.perf_counter() - self.start_times[full_operation]
            self.timings[full_operation].append(elapsed)
            self.counts[full_operation] += 1
            del self.start_times[full_operation]
        self.current_stack.pop()

    @contextmanager
    def time_operation(self, operation: str):
        """Context manager for timing an operation."""
        self.start(operation)
        try:
            yield
        finally:
            self.end(operation)

    def add_timing(self, operation: str, elapsed: float, count: int = 1):
        """Add a timing measurement directly."""
        self.timings[operation].append(elapsed)
        self.counts[operation] = max(self.counts[operation], count)  # Store max count for display

    def get_summary(self) -> Dict[str, Dict[str, float]]:
        """Get summary statistics for all timings."""
        summary = {}
        for operation, times in self.timings.items():
            if times:
                summary[operation] = {
                    "total": sum(times),
                    "count": len(times),
                    "avg": sum(times) / len(times),
                    "min": min(times),
                    "max": max(times),
                }
        return summary

    def print_report(self, title: str = "Parsing Timing Report"):
        """Print a formatted timing report."""
        summary = self.get_summary()
        if not summary:
            logger.info(f"{title}: No timing data collected.")
            return

        # Sort by total time descending
        sorted_ops = sorted(summary.items(), key=lambda x: x[1]["total"], reverse=True)

        logger.info("=" * 100)
        logger.info(f"{title}")
        logger.info("=" * 100)
        logger.info(f"{'Operation':<60} {'Total (s)':<12} {'Count':<8} {'Avg (s)':<12} {'Min (s)':<12} {'Max (s)':<12}")
        logger.info("-" * 100)

        total_time = 0
        for operation, stats in sorted_ops:
            total_time += stats["total"]
            count_display = self.counts.get(operation, stats['count'])
            logger.info(
                f"{operation[:59]:<60} "
                f"{stats['total']:>11.4f} "
                f"{count_display:>7} "
                f"{stats['avg']:>11.4f} "
                f"{stats['min']:>11.4f} "
                f"{stats['max']:>11.4f}"
            )

        logger.info("-" * 100)
        logger.info(f"{'TOTAL':<60} {total_time:>11.4f}")
        logger.info("=" * 100)

    def reset(self):
        """Reset all collected timing data."""
        self.timings.clear()
        self.counts.clear()
        self.current_stack.clear()
        self.start_times.clear()


# Global timing collector instance
_global_collector: Optional[TimingCollector] = None


def get_timing_collector() -> TimingCollector:
    """Get the global timing collector instance."""
    global _global_collector
    if _global_collector is None:
        _global_collector = TimingCollector()
    return _global_collector


def reset_timing_collector():
    """Reset the global timing collector."""
    global _global_collector
    if _global_collector is not None:
        _global_collector.reset()


def print_timing_report(title: str = "Parsing Timing Report"):
    """Print timing report from global collector."""
    collector = get_timing_collector()
    collector.print_report(title)

