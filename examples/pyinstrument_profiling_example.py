#!/usr/bin/env python3
"""
Example demonstrating the usage of PyinstrumentProfilerMixin.

This script shows how to use the PyinstrumentProfilerMixin for profiling
Python code performance using the pyinstrument statistical profiler.
"""

import math
import time

from fusion_bench.mixins.pyinstrument import PyinstrumentProfilerMixin


class ExampleClass(PyinstrumentProfilerMixin):
    """Example class that demonstrates profiling capabilities."""

    def expensive_computation(self, n: int = 1000000):
        """Simulate an expensive computation."""
        result = 0
        for i in range(n):
            result += math.sqrt(i) * math.sin(i)
        return result

    def io_simulation(self, duration: float = 0.5):
        """Simulate I/O operations."""
        time.sleep(duration)

    def run_example(self):
        """Run the example with profiling."""
        print("Running example with profiling...")

        # Example 1: Profile individual operations
        with self.profile("expensive_computation"):
            result1 = self.expensive_computation(500000)

        with self.profile("io_simulation"):
            self.io_simulation(0.2)

        # Example 2: Profile a larger block
        with self.profile("combined_operations"):
            result2 = self.expensive_computation(300000)
            self.io_simulation(0.1)

        # Print the profiling summary
        self.print_profile_summary("=== Profiling Summary ===")

        # Save detailed HTML report
        self.save_profile_report("profile_report.html", format="html")

        print(f"Results: {result1:.2f}, {result2:.2f}")
        print("Profile report saved to 'profile_report.html'")


if __name__ == "__main__":
    example = ExampleClass()
    example.run_example()
