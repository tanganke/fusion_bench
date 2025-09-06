# PyinstrumentProfilerMixin

The `PyinstrumentProfilerMixin` provides statistical profiling capabilities using the [pyinstrument](https://pyinstrument.readthedocs.io/) library. This mixin allows you to easily profile code execution time and identify performance bottlenecks in your applications.

## Installation

First, install pyinstrument:

```bash
pip install pyinstrument
```

## Usage

### Basic Usage

```python
from fusion_bench.mixins import PyinstrumentProfilerMixin

class MyClass(PyinstrumentProfilerMixin):
    def expensive_computation(self):
        # Profile a specific code block
        with self.profile("computation"):
            # Your expensive code here
            result = sum(i**2 for i in range(1000000))
        
        # Print profiling summary
        self.print_profile_summary("Performance Report")
        
        # Save detailed HTML report
        self.save_profile_report("profile.html")
        
        return result
```

### Manual Profiling Control

```python
class MyClass(PyinstrumentProfilerMixin):
    def run_analysis(self):
        # Start profiling manually
        self.start_profile("analysis")
        
        # Your code here
        self.step1()
        self.step2()
        self.step3()
        
        # Stop profiling
        self.stop_profile("analysis")
        
        # Print results
        self.print_profile_summary()
```

### Multiple Output Formats

```python
class MyClass(PyinstrumentProfilerMixin):
    def analyze_performance(self):
        with self.profile("analysis"):
            # Your code here
            pass
        
        # Save in different formats
        self.save_profile_report("report.html", format="html")
        self.save_profile_report("report.txt", format="text")
```

## Implementation Details

- [fusion_bench.mixins.PyinstrumentProfilerMixin][]: The mixin class that provides the profiling functionality.
