"""
Legacy-build-path compatibility shim — Phase D17 packaging repair.

This project's authoritative package metadata lives entirely in
pyproject.toml's `[project]` table (PEP 621) and is read by the modern
PEP 517 `setuptools.build_meta` backend declared in `[build-system]` —
this file does NOT duplicate that as the source of truth, and a normal
`pip install` (isolated or not) against any PEP 621-aware setuptools
(>=61) ignores the name/version passed below entirely in favor of
pyproject.toml's own values (verified directly, Phase D17: mismatched
name/version passed here were silently overridden by pyproject.toml's
real values on setuptools 79/84).

Why this file exists at all, given that: some build front ends invoke
`setup.py egg_info`/`setup.py build` directly (bypassing the PEP 517
`prepare_metadata_for_build_wheel`/`build_wheel` hooks entirely) — most
relevantly, `pip install` with build isolation unavailable/disabled
running against a setuptools version old enough to predate `[project]`
table support (PEP 621 support landed in setuptools 61.0). In exactly
that situation, a pyproject.toml-only project (no setup.py at all) has
no source setuptools can read a name/version from, and setuptools falls
back to its long-standing literal placeholder, "UNKNOWN" — the exact
`UNKNOWN.egg-info`/`UNKNOWN.dist-info` failure this phase's task
describes. A minimal, explicit `setup(name=..., version=...)` call gives
that legacy path a real value to fall back to instead, at zero cost and
zero risk to the modern path (see docs/DPE_V1_PROVIDER_CONTRACT.md's D17
record for the full investigation, including what was and was not able
to be reproduced directly).

Keep `name`/`version` here manually in sync with pyproject.toml's own
`[project].name`/`[project].version` — both are static, hand-edited
values in this repository already (no dynamic/generated version scheme
exists), the same discipline release commits already require of
src/depth_perception_engine/__init__.py's own `__version__`.
"""

from setuptools import setup

setup(name="depth-perception-engine", version="1.1.0")
