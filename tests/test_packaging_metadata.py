"""
Packaging/release metadata guards — Phase D17 (packaging repair; see
docs/DPE_V1_PROVIDER_CONTRACT.md's D17 record).

Guards specifically against the exact class of failure discovered in
D17: installing this package via a build path that cannot derive a real
name/version from pyproject.toml's `[project]` table (observed
externally as `pip install git+https://...` producing
`UNKNOWN.egg-info`/`UNKNOWN.dist-info`, `pip show` finding nothing, and
`import depth_perception_engine` failing).

Deliberately NOT a network/VCS-install test (Task's own explicit
"prefer a lightweight metadata/import/build check rather than turning
normal unit tests into expensive network-install tests" instruction) —
every test here is static-file-parsing or a local, no-network wheel
build using whatever setuptools/wheel are already importable in the
current interpreter.
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
