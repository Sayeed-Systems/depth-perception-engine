"""
Packaging/release metadata guards — Phase D17 (packaging repair) and
Phase D18 (v1.1.1 packaging repair; see docs/DPE_V1_PROVIDER_CONTRACT.md's
D17/D18 records).

Guards against two distinct, proven failure classes:

D17 — installing via a build path that cannot derive a real name/version
from pyproject.toml's `[project]` table (observed externally as
`pip install git+https://...` producing `UNKNOWN.egg-info`/
`UNKNOWN.dist-info`, `pip show` finding nothing, `import
depth_perception_engine` failing).

D18 — a real `pip install`/`pip wheel` under genuine PEP 517 build
ISOLATION on Python 3.10 (Ubuntu 22.04 dist-packages layout) silently
builds a wheel with correct name/version metadata but ZERO actual code —
no error, no warning, `pip` reports success, `import
depth_perception_engine` then fails with a plain `ModuleNotFoundError`.
Proven NOT reproducible via `--no-build-isolation` (any setuptools
version) — which is exactly why `TestRealWheelBuildProducesCorrectMetadata`
below, D17's own "real build" check, did not catch it: it deliberately
uses `--no-build-isolation`. `TestIsolatedBuildProducesNonEmptyWheel`
closes that gap with a genuinely isolated build check.

Deliberately NOT expensive network/VCS-install tests as the primary
mechanism (Task's own explicit "prefer a lightweight metadata/import/
build check rather than turning normal unit tests into expensive
network-install tests" instruction) — most tests here are static-file-
parsing or a local, no-network wheel build using whatever setuptools/
wheel are already importable in the current interpreter.
`TestIsolatedBuildProducesNonEmptyWheel` is the one exception: a genuine
isolated build necessarily fetches its own setuptools/wheel from PyPI
(that fetch is the exact mechanism under test), so it needs network and
skips cleanly, not fails, when network is unavailable.
"""

import ast
import os
import re
import subprocess
import sys
import tempfile

import pytest

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
_PYPROJECT_PATH = os.path.join(_REPO_ROOT, "pyproject.toml")
_SETUP_PY_PATH = os.path.join(_REPO_ROOT, "setup.py")
_INIT_PATH = os.path.join(_REPO_ROOT, "src", "depth_perception_engine", "__init__.py")

_EXPECTED_DISTRIBUTION_NAME = "depth-perception-engine"
_EXPECTED_IMPORT_NAME = "depth_perception_engine"


def _load_pyproject():
    try:
        import tomllib
    except ModuleNotFoundError:
        pytest.skip("tomllib not available on this Python version (<3.11) and no fallback TOML parser is a declared dependency")
    with open(_PYPROJECT_PATH, "rb") as f:
        return tomllib.load(f)


def _init_version() -> str:
    with open(_INIT_PATH, "r", encoding="utf-8") as f:
        source = f.read()
    match = re.search(r'^__version__\s*=\s*["\']([^"\']+)["\']', source, re.MULTILINE)
    assert match, "src/depth_perception_engine/__init__.py has no __version__ assignment"
    return match.group(1)


def _setup_py_kwargs():
    """Parse setup.py's own setup(...) call arguments via AST — never
    executes setup.py (which would trigger a real build command)."""
    with open(_SETUP_PY_PATH, "r", encoding="utf-8") as f:
        tree = ast.parse(f.read(), filename=_SETUP_PY_PATH)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup":
            return {kw.arg: kw.value.value for kw in node.keywords if isinstance(kw.value, ast.Constant)}
    raise AssertionError("setup.py contains no setup(...) call")


class TestDistributionAndImportNames:
    def test_pyproject_declares_the_expected_distribution_name(self):
        data = _load_pyproject()
        assert data["project"]["name"] == _EXPECTED_DISTRIBUTION_NAME

    def test_import_package_directory_matches_the_expected_import_name(self):
        assert os.path.isdir(os.path.join(_REPO_ROOT, "src", _EXPECTED_IMPORT_NAME))


class TestSetupPyCompatibilityShimExists:
    """Phase D17: pyproject.toml-only projects (no setup.py at all) fall
    back to setuptools' literal "UNKNOWN" name/version placeholder under
    a build path that cannot read [project] (see this file's own module
    docstring). setup.py's presence — and its explicit name/version
    kwargs, which a PEP-621-aware setuptools silently ignores in favor
    of pyproject.toml's own values (verified directly in D17) — is the
    fix. This test guards against a future edit accidentally deleting
    setup.py without understanding why it exists."""

    def test_setup_py_exists(self):
        assert os.path.isfile(_SETUP_PY_PATH), (
            "setup.py is missing — this is the Phase D17 legacy-build-path "
            "compatibility shim; see its own module docstring before removing it."
        )

    def test_setup_py_declares_a_real_name_and_version(self):
        kwargs = _setup_py_kwargs()
        assert kwargs.get("name") == _EXPECTED_DISTRIBUTION_NAME
        assert kwargs.get("name") != "UNKNOWN"
        assert kwargs.get("version"), "setup.py's setup() call has no version kwarg"
        assert kwargs.get("version") != "UNKNOWN"


class TestSetupPyDeclaresPackagesExplicitly:
    """Phase D18: package discovery must be declared imperatively in
    setup.py (`packages=find_packages(...)`, `package_dir=...`), NOT via
    pyproject.toml's `[tool.setuptools.packages.find]` TOML auto-discovery
    — that mechanism was proven to silently resolve to zero packages under
    real PEP 517 build isolation on Python 3.10 (see this file's own
    module docstring and setup.py's own docstring for the full
    investigation). `_setup_py_kwargs()` only captures `ast.Constant`
    keyword values (see its own docstring) — `packages`/`package_dir` are
    a function call and a dict, not constants, so this test parses the AST
    directly instead."""

    def test_setup_py_call_has_packages_and_package_dir_keywords(self):
        with open(_SETUP_PY_PATH, "r", encoding="utf-8") as f:
            tree = ast.parse(f.read(), filename=_SETUP_PY_PATH)
        setup_call = next(
            node for node in ast.walk(tree)
            if isinstance(node, ast.Call) and getattr(node.func, "id", None) == "setup"
        )
        keyword_names = {kw.arg for kw in setup_call.keywords}
        assert "packages" in keyword_names, (
            "setup.py's setup() call has no packages= keyword — this is the "
            "Phase D18 fix for the empty-isolated-wheel defect; see setup.py's "
            "own module docstring before removing it."
        )
        assert "package_dir" in keyword_names, (
            "setup.py's setup() call has no package_dir= keyword — required "
            "alongside packages= for the src/ layout to resolve correctly."
        )

    def test_pyproject_toml_does_not_redeclare_packages_find(self):
        # A [tool.setuptools.packages.find] section coexisting with
        # setup.py's own explicit packages= would reintroduce exactly the
        # ambiguity this phase's fix removes — guard against silently
        # re-adding it. Checked via the parsed TOML structure, not a raw
        # substring match, so this doesn't false-positive on prose (e.g.
        # this file's own docstrings/comments) mentioning the section name.
        data = _load_pyproject()
        find_section = data.get("tool", {}).get("setuptools", {}).get("packages", {}).get("find")
        assert find_section is None, (
            "pyproject.toml has a [tool.setuptools.packages.find] section — "
            "this was removed in Phase D18 because it silently produces an "
            "empty wheel under real build isolation on Python 3.10; package "
            "discovery must stay declared only in setup.py. See "
            "docs/DPE_V1_PROVIDER_CONTRACT.md's D18 record before re-adding it."
        )


class TestVersionConsistency:
    """The exact three places this repository hand-maintains a version
    string must always agree — a silent drift here is exactly the kind
    of defect that makes a release's own metadata untrustworthy."""

    def test_pyproject_init_and_setup_py_versions_all_agree(self):
        pyproject_version = _load_pyproject()["project"]["version"]
        init_version = _init_version()
        setup_py_version = _setup_py_kwargs().get("version")

        assert pyproject_version == init_version, (
            f"pyproject.toml version ({pyproject_version!r}) != "
            f"__init__.py __version__ ({init_version!r})"
        )
        assert pyproject_version == setup_py_version, (
            f"pyproject.toml version ({pyproject_version!r}) != "
            f"setup.py's setup() version kwarg ({setup_py_version!r})"
        )

    def test_version_is_not_a_placeholder(self):
        for version in (_load_pyproject()["project"]["version"], _init_version()):
            assert version not in ("0.0.0", "UNKNOWN", ""), f"placeholder version found: {version!r}"


class TestLicenseFormIsTheSetuptoolsCompatibleForm:
    """Phase D17: the plain SPDX-string license form (`license = "MIT"`)
    was tried and reverted after being proven to HARD-ERROR on
    setuptools 70 ("`project.license` must be valid exactly by one
    definition"). Guards against silently "modernizing" this back to
    the string form without re-verifying against the actual minimum
    setuptools this project needs to keep working — see this file's own
    pyproject.toml comment and docs/DPE_V1_PROVIDER_CONTRACT.md's D17
    record."""

    def test_license_is_the_table_form_not_a_bare_string(self):
        data = _load_pyproject()
        license_value = data["project"]["license"]
        assert isinstance(license_value, dict) and "text" in license_value, (
            f"project.license changed to {license_value!r} — re-read "
            "docs/DPE_V1_PROVIDER_CONTRACT.md's D17 record before changing "
            "this; the plain SPDX string form previously broke setuptools 70."
        )


class TestRealWheelBuildProducesCorrectMetadata:
    """The one real (not merely static-parsed) build check this file
    performs — deliberately lightweight: uses --no-build-isolation
    against whatever setuptools/wheel are already importable in the
    CURRENT interpreter (no fresh isolated environment, no network
    fetch), skipped cleanly if they aren't installed here rather than
    failing the whole suite over an environment gap unrelated to DPE's
    own source correctness."""

    def test_wheel_build_has_no_unknown_metadata(self):
        try:
            import setuptools  # noqa: F401
            import wheel  # noqa: F401
        except ModuleNotFoundError:
            pytest.skip("setuptools/wheel not importable in the current interpreter — install them to exercise this check")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", _REPO_ROOT,
                 "--no-deps", "--no-build-isolation", "-w", tmp_dir],
                capture_output=True, text=True, timeout=120,
            )
            assert result.returncode == 0, (
                f"wheel build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            built = [f for f in os.listdir(tmp_dir) if f.endswith(".whl")]
            assert len(built) == 1, f"expected exactly one wheel, found: {built}"
            wheel_name = built[0]

            assert "UNKNOWN" not in wheel_name, f"wheel filename contains UNKNOWN: {wheel_name}"
            assert wheel_name.startswith(f"{_EXPECTED_IMPORT_NAME}-"), (
                f"wheel filename {wheel_name!r} does not start with {_EXPECTED_IMPORT_NAME!r}"
            )

            import zipfile
            with zipfile.ZipFile(os.path.join(tmp_dir, wheel_name)) as z:
                dist_info = next(n for n in z.namelist() if n.endswith(".dist-info/METADATA"))
                assert "UNKNOWN" not in dist_info, f"dist-info directory contains UNKNOWN: {dist_info}"
                metadata = z.read(dist_info).decode()
                assert f"Name: {_EXPECTED_DISTRIBUTION_NAME}" in metadata
                assert "Name: UNKNOWN" not in metadata
                assert "Version: UNKNOWN" not in metadata
                assert f"{_EXPECTED_IMPORT_NAME}/__init__.py" in z.namelist()


class TestIsolatedBuildProducesNonEmptyWheel:
    """Phase D18: the actual check that catches the empty-wheel defect —
    a GENUINELY isolated build (no --no-build-isolation), which is exactly
    what pip does for a plain `pip install depth-perception-engine` or
    `pip install git+https://...`, and exactly the invocation
    TestRealWheelBuildProducesCorrectMetadata's own --no-build-isolation
    choice does not exercise.

    Needs network (an isolated build fetches its own setuptools/wheel from
    PyPI, per pyproject.toml's [build-system] requires) — skips cleanly,
    not fails, if a quick reachability probe to pypi.org fails, matching
    this file's own "environment gap is not DPE's fault" philosophy.
    """

    def test_isolated_pip_wheel_build_contains_real_code(self):
        import socket
        try:
            socket.create_connection(("pypi.org", 443), timeout=3).close()
        except OSError:
            pytest.skip("no network access to pypi.org — cannot exercise a genuinely isolated build")

        with tempfile.TemporaryDirectory() as tmp_dir:
            result = subprocess.run(
                [sys.executable, "-m", "pip", "wheel", _REPO_ROOT, "--no-deps", "-w", tmp_dir],
                capture_output=True, text=True, timeout=300,
            )
            assert result.returncode == 0, (
                f"isolated wheel build failed:\nSTDOUT:\n{result.stdout}\nSTDERR:\n{result.stderr}"
            )
            built = [f for f in os.listdir(tmp_dir) if f.endswith(".whl")]
            assert len(built) == 1, f"expected exactly one wheel, found: {built}"

            import zipfile
            with zipfile.ZipFile(os.path.join(tmp_dir, built[0])) as z:
                names = z.namelist()
                assert f"{_EXPECTED_IMPORT_NAME}/__init__.py" in names, (
                    f"isolated build produced a wheel with no {_EXPECTED_IMPORT_NAME}/"
                    f"__init__.py — this is exactly the Phase D18 empty-wheel defect. "
                    f"Wheel contents: {names}"
                )
                # A handful of real submodules, not just __init__.py — the D18
                # defect produced dist-info-only wheels; this also catches a
                # PARTIAL discovery regression, not only a total one.
                for expected_module in (
                    f"{_EXPECTED_IMPORT_NAME}/geometry/__init__.py",
                    f"{_EXPECTED_IMPORT_NAME}/pipeline/pipeline.py",
                    f"{_EXPECTED_IMPORT_NAME}/obstacles/threat_assessment.py",
                ):
                    assert expected_module in names, (
                        f"isolated build wheel is missing {expected_module} — "
                        f"partial package-discovery regression. Wheel contents: {names}"
                    )
                py_file_count = sum(1 for n in names if n.startswith(f"{_EXPECTED_IMPORT_NAME}/") and n.endswith(".py"))
                assert py_file_count >= 50, (
                    f"isolated build wheel has only {py_file_count} .py files under "
                    f"{_EXPECTED_IMPORT_NAME}/ — expected 50+; looks like a partial "
                    f"discovery regression, not a healthy build."
                )
