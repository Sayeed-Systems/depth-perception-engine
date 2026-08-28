"""
Legacy-build-path compatibility shim AND authoritative package-discovery
declaration — Phase D17 packaging repair, superseded/extended by Phase D18
(v1.1.1 packaging repair).

`name`/`version` passed to `setup()` below remain inert on the modern PEP
621 path exactly as D17 originally verified: PEP-621-aware setuptools
(tested 79/84) silently ignores them in favor of pyproject.toml's own
`[project]` table values even when deliberately mismatched. That part of
this file's original purpose (a legacy-build-path fallback for a build tool
old enough to run `setup.py egg_info` directly, bypassing `[project]`
entirely) is unchanged — see D17's own record for that investigation.

`packages`/`package_dir` below are NOT inert, and are the actual fix for a
newly discovered, more severe defect (Phase D18): v1.1.0 declared package
discovery declaratively, in pyproject.toml's `[tool.setuptools.packages.find]
where = ["src"]`, with no explicit `packages=`/`package_dir=` anywhere in
this file. That TOML-driven auto-discovery mechanism was proven, via direct
reproduction, to silently resolve to ZERO packages under real `pip install`/
`pip wheel` PEP 517 build isolation on Python 3.10 (Ubuntu 22.04
dist-packages layout — the exact environment MP01's own `mp01_ros2`
container uses) — the build reports success, `dist-info` metadata
(name/version/dependencies) is completely correct, but the wheel contains
no actual `depth_perception_engine` code at all, with no error or warning
surfaced anywhere in the process. `import depth_perception_engine` then
fails with a plain `ModuleNotFoundError` despite `pip` having reported
"Successfully installed".

Proven NOT reproducible via `pip install --no-build-isolation` (regardless
of which setuptools version — 79.0.1 and 84.0.0 both tested — is present in
the calling environment), and NOT reproducible via a direct, manual
`setuptools.build_meta.build_wheel()` call using the exact same
overlay-setuptools/PYTHONPATH pip's own isolation mechanism installs and
sets — the failure is specific to pip's real, actual isolated-subprocess
build invocation, not to any setuptools version in isolation, an
auto-discovery `include=` pattern (tested, did not fix it), or a "minimal
bare environment" effect (a from-scratch venv with only setuptools/wheel,
`--no-build-isolation`, built correctly). The exact internal pip/setuptools
mechanism was not traced further than this (see D18's own record for the
full elimination sequence) — what IS proven, directly, is that moving
package discovery to this file's own imperative `find_packages()` call,
executed as ordinary Python code inside whatever build hook setuptools
runs, produces the correct, fully-populated wheel under the exact same
real isolated-build invocation that previously produced an empty one.

Keep `packages`/`package_dir` here as the single source of truth for what
ships in the wheel — do not reintroduce a `[tool.setuptools.packages.find]`
section in pyproject.toml without re-verifying this exact failure mode
first (a real, isolated `pip wheel` build on Python 3.10, not
`--no-build-isolation` — see `tests/test_packaging_metadata.py`'s
`TestIsolatedBuildProducesNonEmptyWheel`, which reproduces the check that
actually catches this).

Keep `name`/`version` here manually in sync with pyproject.toml's own
`[project].name`/`[project].version`, and `packages`'s own module list in
sync with whatever `find_packages(where="src")` returns (automatic —
recomputed at build time, never hand-listed) — both are static, hand-edited
values in this repository already (no dynamic/generated version scheme
exists), the same discipline release commits already require of
src/depth_perception_engine/__init__.py's own `__version__`.
"""

from setuptools import find_packages, setup

setup(
    name="depth-perception-engine",
    version="1.2.0",
    package_dir={"": "src"},
    packages=find_packages(where="src"),
)
