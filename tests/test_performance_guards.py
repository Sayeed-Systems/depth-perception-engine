"""
Performance regression guards — Level 3, Phase E6 (Task 11).

Deliberately NOT millisecond-precision assertions (Task 11: "do not add
brittle CI tests requiring exact millisecond latency" — benchmark
tooling lives separately, under examples/benchmark_*.py, and is run
manually/on demand, not as part of this correctness suite). Instead:
a structural, algorithmic-invariant check — every Level 3 geometry
producer is fully vectorized (NumPy/OpenCV array operations only), with
zero Python-level `for`/`while` loops anywhere in their implementation.
An accidental per-pixel Python loop introduced in a future change to any
of these files would silently reintroduce the exact class of performance
regression E2-E5 each individually verified their own implementation
avoided (see each module's own docstring) — this test catches that
class of regression structurally, without pinning any specific timing
number.
"""

import ast
import os

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_GEOMETRY_DIR = os.path.join(_REPO_ROOT, "src", "depth_perception_engine", "geometry")

_VECTORIZED_MODULES = [
    "point_cloud_builder.py",
    "rigid_transform.py",
    "obstacle_extractor.py",
    "free_space.py",
    "geometry_metrics.py",
]


def test_no_for_or_while_loops_in_vectorized_geometry_modules():
    offenders = []
    for filename in _VECTORIZED_MODULES:
        path = os.path.join(_GEOMETRY_DIR, filename)
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, (ast.For, ast.While)):
                offenders.append(f"{filename}:{node.lineno}")

    assert not offenders, (
        "Found a Python for/while loop in a module documented as fully "
        f"vectorized (no per-pixel loop): {offenders}. If this loop is "
        "genuinely necessary, update this test deliberately and document "
        "why vectorization is not possible here, rather than silently "
        "letting a per-pixel loop back in."
    )


def test_benchmark_tooling_lives_outside_the_test_suite():
    """Task 11: "keep benchmark tooling separate from correctness
    tests" — verified structurally: every examples/benchmark_*.py file
    exists under examples/, not tests/, and no tests/ file imports one as
    a module (which would accidentally couple correctness-test pass/fail
    to wall-clock timing)."""
    examples_dir = os.path.join(_REPO_ROOT, "examples")
    tests_dir = os.path.join(_REPO_ROOT, "tests")

    benchmark_files = [f for f in os.listdir(examples_dir) if f.startswith("benchmark_") and f.endswith(".py")]
    assert len(benchmark_files) >= 5, (
        f"expected at least the E2-E6 benchmark scripts to exist under examples/, found: {benchmark_files}"
    )

    for filename in os.listdir(tests_dir):
        if not filename.endswith(".py") or filename == "test_performance_guards.py":
            continue  # this file's own docstring/comments legitimately discuss benchmark_* by name
        with open(os.path.join(tests_dir, filename), "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=filename)
        for node in ast.walk(tree):
            names = []
            if isinstance(node, ast.Import):
                names = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom) and node.module:
                names = [node.module]
            assert not any("benchmark_" in name for name in names), (
                f"tests/{filename} imports a benchmark_* module — benchmark tooling must stay "
                "out of the correctness test suite's import graph."
            )
