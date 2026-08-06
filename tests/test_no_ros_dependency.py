"""
Proves depth_perception_engine has no ROS dependency and no hardware/GUI
coupling — the core requirement of this refactor.

Two independent checks:

1. Real absence: this test environment has rclpy, sensor_msgs, and
   cv_bridge genuinely NOT installed (see the assertions below) — so if any
   module under depth_perception_engine imported one of them, merely
   importing the package in test_imports.py would already fail with
   ImportError. That already happening successfully is itself evidence.

2. Static source scan: catches forbidden imports/calls even if they're
   deferred inside a function body (so they wouldn't trip at package-import
   time) or behind a try/except ImportError.
"""

import ast
import importlib.util
import os

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_SRC_DIR = os.path.join(_REPO_ROOT, "src", "depth_perception_engine")

_FORBIDDEN_MODULES = {"rclpy", "sensor_msgs", "cv_bridge"}
_FORBIDDEN_CV2_CALLS = {"imshow", "waitKey", "namedWindow", "VideoCapture", "destroyAllWindows"}

# Set by `source install/setup.bash` (or any sourced ROS 2 overlay) — reliable
# signal that rclpy/sensor_msgs/cv_bridge are legitimately on sys.path for
# reasons that have nothing to do with this library importing them (this
# package is now vendored into an ament/colcon ROS workspace precisely so
# `colcon test` can run it there — see package.xml's description). The
# "real absence" check below is only a meaningful signal in a bare, non-ROS
# environment (e.g. `pip install -e .[test] && pytest tests/` outside any
# ROS setup); it is not this library's job to be un-importable ROS itself.
_RUNNING_IN_SOURCED_ROS_ENV = "AMENT_PREFIX_PATH" in os.environ


def _all_source_files():
    for dirpath, _dirnames, filenames in os.walk(_SRC_DIR):
        for filename in filenames:
            if filename.endswith(".py"):
                yield os.path.join(dirpath, filename)


@pytest.mark.parametrize("module_name", sorted(_FORBIDDEN_MODULES))
@pytest.mark.skipif(
    _RUNNING_IN_SOURCED_ROS_ENV,
    reason=(
        "Running inside a sourced ROS 2 environment (AMENT_PREFIX_PATH set) "
        "— rclpy/sensor_msgs/cv_bridge are legitimately installed system-wide "
        "here regardless of this library's own imports. This check is only "
        "meaningful in a bare, non-ROS test environment; the static AST scan "
        "in test_no_forbidden_imports_anywhere_in_the_library below is what "
        "actually proves this library doesn't import them, and still runs "
        "unconditionally in every environment."
    ),
)
def test_forbidden_ros_modules_are_genuinely_not_installed(module_name):
    """Sanity check on the test environment itself: without this being true,
    check #1 in this file's docstring proves nothing."""
    assert importlib.util.find_spec(module_name) is None, (
        f"{module_name} is installed in this environment — "
        "the 'real absence' import check is no longer meaningful here."
    )


def test_no_forbidden_imports_anywhere_in_the_library():
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
            hit = names & _FORBIDDEN_MODULES
            if hit:
                offenders.append(f"{os.path.relpath(path, _REPO_ROOT)}: imports {hit}")

    assert not offenders, "Forbidden ROS imports found:\n" + "\n".join(offenders)


def test_no_gui_or_hardware_capture_calls_in_the_library():
    offenders = []
    for path in _all_source_files():
        with open(path, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=path)
        for node in ast.walk(tree):
            if isinstance(node, ast.Attribute) and node.attr in _FORBIDDEN_CV2_CALLS:
                offenders.append(
                    f"{os.path.relpath(path, _REPO_ROOT)}:{node.lineno}: .{node.attr}(...)"
                )

    assert not offenders, "Forbidden GUI/camera-capture calls found:\n" + "\n".join(offenders)
