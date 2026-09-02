"""
D1 — DPE SYNCHRONOUS EXECUTION + PERFORMANCE AUDIT (clean rerun).

Benchmark/audit tooling only. Nothing in this package is imported by
depth_perception_engine production code, and nothing here modifies
production behaviour: every measurement drives the already-shipped public
API (or re-drives an already-shipped stage function with the exact
arguments pipeline.py itself passes it) and reads the pipeline's own
existing DEBUG-level stage instrumentation.

No threads, queues, futures, multiprocessing, async, or CUDA appear
anywhere in this package — see the D1 report's section 25.
"""
