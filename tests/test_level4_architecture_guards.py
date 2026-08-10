"""
Level 4 (Phase E1) architecture guards.

Statically scans src/depth_perception_engine/ for concepts this phase's
own instructions explicitly forbid from ever appearing in core perception
code — mirrors test_no_ros_dependency.py's existing AST/text-scan style
rather than inventing a new checking approach.

Two independent concerns:
1. Zero dependency on state_estimation_engine, in either direction, and
   zero platform/vehicle-class concept (aerial/ground/marine/rover/boat)
   anywhere in core logic — depth_perception_engine must stay agent-
   agnostic (docs/LEVEL4_ARCHITECTURE.md section 2/3).
2. No simulation-specific branching pattern (docs/LEVEL4_SIMULATED_IMU.md)
   — a MotionHint must be indistinguishable whether it came from a
   simulated or real producer.

Deliberately AST-identifier-based, not raw-text-based: this phase's own
instructions say "documentation explaining forbidden dependencies is
fine... [c]ore implementation must remain clean" — a docstring or comment
*mentioning* a forbidden word while explaining why it's absent (e.g.
temporal/types.py's own docstring explaining there is no `is_simulated`
field) must not trip these guards. Only identifiers actually used as code
— names, function/class defs, parameters, attribute access — are checked.
"""

import ast
import os
import re

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src", "depth_perception_engine")

_FORBIDDEN_WORDS = [
    "state_estimation_engine",
    "aerial",
    "drone",
    "ground",
    "marine",
    "rover",
    "boat",
    "mavlink",
    "pixhawk",
]
_FORBIDDEN_WORD_PATTERN = re.compile(
    r"(?:^|_)(" + "|".join(_FORBIDDEN_WORDS) + r")(?:_|$)", re.IGNORECASE,
)

_SIMULATION_WORDS = ["simulated_imu", "real_imu", "is_simulated", "simulation_source"]
_SIMULATION_PATTERN = re.compile(
    r"(?:^|_)(" + "|".join(_SIMULATION_WORDS) + r")(?:_|$)", re.IGNORECASE,
)


def _all_source_files():
    for dirpath, _dirnames, filenames in os.walk(_SRC_DIR):
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


def _code_identifiers(tree):
    """Every identifier actually used as code — variable/attribute/
    parameter/function/class names — excluding string/docstring contents
    entirely."""
    for node in ast.walk(tree):
        if isinstance(node, ast.Name):
            yield node.id, node.lineno
        elif isinstance(node, ast.Attribute):
            yield node.attr, node.lineno
        elif isinstance(node, ast.arg):
            yield node.arg, node.lineno
        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef, ast.ClassDef)):
            yield node.name, node.lineno
        elif isinstance(node, ast.keyword) and node.arg is not None:
            yield node.arg, node.lineno


def _scan(pattern):
    offenders = []
    for path in _all_source_files():
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for identifier, lineno in _code_identifiers(tree):
            if pattern.search(identifier):
                offenders.append(f"{os.path.relpath(path, _REPO_ROOT)}:{lineno}: identifier {identifier!r}")
    return offenders


def test_no_forbidden_platform_or_state_estimation_engine_terms_in_core_source():
    offenders = _scan(_FORBIDDEN_WORD_PATTERN)
    assert not offenders, (
        "depth_perception_engine must stay platform-agnostic and have zero "
        "dependency on state_estimation_engine (docs/LEVEL4_ARCHITECTURE.md "
        "section 3) — forbidden identifier(s) found in actual code (not "
        "docstrings/comments):\n" + "\n".join(offenders)
    )


def test_no_simulation_specific_identifiers_in_core_source():
    offenders = _scan(_SIMULATION_PATTERN)
    assert not offenders, (
        "Core perception code must never branch on simulated-vs-real motion "
        "input source — see docs/LEVEL4_SIMULATED_IMU.md:\n" + "\n".join(offenders)
    )


def test_no_string_comparison_against_simulation_source_literals():
    """AST-based check for `x == "simulation"` / `x == "simulated"` style
    comparisons specifically, catching the pattern even if no identifier
    named e.g. `simulated_imu` exists but a raw string literal comparison
    does."""
    offenders = []
    literals = {"simulation", "simulated", "real"}
    for path in _all_source_files():
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Compare):
                for comparator in node.comparators:
                    if isinstance(comparator, ast.Constant) and isinstance(comparator.value, str):
                        if comparator.value.lower() in literals:
                            offenders.append(
                                f"{os.path.relpath(path, _REPO_ROOT)}:{node.lineno}: "
                                f"comparison against {comparator.value!r}"
                            )

    assert not offenders, (
        "Found a source-specific comparison — depth_perception_engine must "
        "not branch on whether a motion input is simulated or real:\n"
        + "\n".join(offenders)
    )


def test_no_dependency_on_state_estimation_engine_package():
    """Static import scan, mirroring test_no_ros_dependency.py's own
    approach exactly — catches an import even if deferred inside a
    function body or behind try/except ImportError."""
    offenders = []
    for path in _all_source_files():
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = {alias.name.split(".")[0] for alias in node.names}
            elif isinstance(node, ast.ImportFrom):
                names = {node.module.split(".")[0]} if node.module else set()
            else:
                continue
            if "state_estimation_engine" in names:
                offenders.append(f"{os.path.relpath(path, _REPO_ROOT)}: imports state_estimation_engine")

    assert not offenders, "Forbidden state_estimation_engine import found:\n" + "\n".join(offenders)
