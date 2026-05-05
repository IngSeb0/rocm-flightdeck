"""Benchmark methodology fixture.

Benchmarks not executed in this MVP.
Run this file on AMD Developer Cloud / MI300X to collect real metrics.
"""


METRICS_TO_COLLECT = [
    "tokens/sec",
    "latency",
    "VRAM usage",
    "cold start time",
    "test pass rate",
]
