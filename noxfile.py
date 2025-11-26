"""Nox config."""

import os
import pathlib
import shutil

import nox
import nox_uv

# Options to modify nox behaviour
nox.options.default_venv_backend = "uv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = [
    "lint",
    "tests",
]

ALL_PYTHON = [
    "3.10",
    "3.11",
    "3.12",
    "3.13",
    "3.14",
]
ARRAY_BACKENDS = {
    "array_api_strict": "array-api-strict>=2",
    "jax": "jax>=0.4.32",
}
BENCH_TESTS_LOC = pathlib.Path("tests/benchmarks")
GLASS_REPO_URL = "https://github.com/glass-dev/glass"
SHARED_BENCHMARK_FLAGS = [
    "--benchmark-columns=mean,stddev,rounds",
    "--benchmark-sort=name",
    "--benchmark-timer=time.process_time",
]


def _check_revision_count(
    session_posargs: list[str],
    *,
    expected_count: int,
) -> None:
    """Check that the correct number of revisions have been provided.

    Parameters
    ----------
    session_posargs
        The positional arguments passed to the session.
    expected_count
        The expected number of revisions.

    Raises
    ------
    ValueError
        If no revisions are provided.
    ValueError
        If the number of provided revisions does not match the expected count.
    """
    if not session_posargs:
        msg = f"{expected_count} revision(s) not provided"
        raise ValueError(msg)

    if len(session_posargs) != expected_count:
        msg = (
            f"Incorrect number of revisions provided ({len(session_posargs)}), "
            f"expected {expected_count}"
        )
        raise ValueError(msg)


@nox_uv.session(
    uv_no_install_project=True,
    uv_only_groups=["lint"],
)
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.run("pre-commit", "run", "--all-files", *session.posargs)


def _setup_array_backend(session: nox.Session) -> None:
    """Installs the requested array_backend."""
    array_backend = os.environ.get("ARRAY_BACKEND")
    if array_backend == "array_api_strict":
        session.install(ARRAY_BACKENDS["array_api_strict"])
    elif array_backend == "jax":
        session.install(ARRAY_BACKENDS["jax"])
    elif array_backend == "all":
        session.install(*ARRAY_BACKENDS.values())


@nox_uv.session(
    python=ALL_PYTHON,
    uv_groups=["test"],
)
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    _setup_array_backend(session)
    session.run("pytest", *session.posargs)


@nox_uv.session(
    python=ALL_PYTHON,
    uv_groups=["test"],
)
def coverage(session: nox.Session) -> None:
    """Run tests and compute coverage for the core tests."""
    _setup_array_backend(session)
    session.run(
        "pytest",
        "--cov",
        *session.posargs,
        env=os.environ,
    )


@nox_uv.session(
    python=ALL_PYTHON,
    uv_groups=["test"],
)
def coverage_benchmarks(session: nox.Session) -> None:
    """Run tests and compute coverage for the benchmark tests."""
    _setup_array_backend(session)
    session.run(
        "pytest",
        BENCH_TESTS_LOC,
        "--cov",
        *SHARED_BENCHMARK_FLAGS,
        *session.posargs,
        env=os.environ,
    )


@nox_uv.session(
    python=ALL_PYTHON,
    uv_groups=["doctest"],
    uv_no_install_project=True,
)
def doctests(session: nox.Session) -> None:
    """Run the doctests."""
    session.posargs.extend(
        [
            "--doctest-plus",
            "--doctest-plus-generate-diff=overwrite",
            "glass",
        ],
    )
    session.run("pytest", *session.posargs)


@nox_uv.session(uv_extras=["examples"])
def examples(session: nox.Session) -> None:
    """Run the example notebooks. Pass "html" to build html."""
    if session.posargs:
        if "html" in session.posargs:
            session.log("Generating HTML for the example notebooks")
            session.run(
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--embed-images",
                "examples/**/*.ipynb",
            )
        else:
            session.log("Unsupported argument to examples")
    else:
        session.run(
            "jupyter",
            "execute",
            "--inplace",
            *pathlib.Path().glob("examples/**/*.ipynb"),
            *session.posargs,
        )


@nox_uv.session(uv_groups=["docs"])
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.chdir("docs")
    session.run(
        "sphinx-build",
        "-M",
        "html",
        ".",
        "_build",
        "--fail-on-warning",
    )

    if session.posargs:
        if "serve" in session.posargs:
            port = 8001

            session.log(
                f"Launching docs at http://localhost:{port}/ - use Ctrl-C to quit",
            )
            session.run("python", "-m", "http.server", f"{port}", "-d", "_build/html")
        else:
            session.log("Unsupported argument to docs")


@nox_uv.session(
    uv_no_install_project=True,
    uv_only_groups=["build"],
)
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.run("python", "-m", "build")


@nox_uv.session
def version(session: nox.Session) -> None:
    """
    Check the current version of the package.

    The intent of this check is to ensure that the package
    is installed without any additional dependencies
    through optional dependencies nor dependency groups.
    """
    session.run("python", "-c", "import glass; print(glass.__version__)")


@nox_uv.session(
    uv_no_install_project=True,
    uv_only_groups=["test"],
)
def benchmarks(session: nox.Session) -> None:
    """
    Run the benchmark test for a specific revision.

    Note it is not possible to pass extra options to pytest.
    """
    _check_revision_count(session.posargs, expected_count=1)
    revision = session.posargs[0]

    # overwrite current package with specified revision
    session.install(f"git+{GLASS_REPO_URL}@{revision}")
    session.run("pytest", BENCH_TESTS_LOC)


@nox_uv.session(
    uv_no_install_project=True,
    uv_only_groups=["test"],
)
def regression_tests(session: nox.Session) -> None:
    """
    Run regression benchmark tests between two revisions.

    Note it is not possible to pass extra options to pytest.
    """
    _check_revision_count(session.posargs, expected_count=2)
    before_revision, after_revision = session.posargs

    _setup_array_backend(session)

    # make sure benchmark directory is clean
    benchmark_dir = pathlib.Path(".benchmarks")
    if benchmark_dir.exists():
        session.log(f"Deleting previous benchmark directory: {benchmark_dir}")
        shutil.rmtree(benchmark_dir)

    session.log(f"Generating prior benchmark from revision {before_revision}")
    session.install(f"git+{GLASS_REPO_URL}@{before_revision}")
    session.run(
        "pytest",
        BENCH_TESTS_LOC,
        "--benchmark-autosave",
        *SHARED_BENCHMARK_FLAGS,
    )

    session.log(f"Comparing {before_revision} benchmark to revision {after_revision}")
    session.install(f"git+{GLASS_REPO_URL}@{after_revision}")
    session.log("Running stable regression tests")
    session.run(
        "pytest",
        BENCH_TESTS_LOC,
        "-m",
        "stable",
        "--benchmark-compare=0001",
        "--benchmark-compare-fail=mean:5%",
        *SHARED_BENCHMARK_FLAGS,
    )

    session.log("Running unstable regression tests")
    session.run(
        "pytest",
        BENCH_TESTS_LOC,
        "-m",
        "unstable",
        "--benchmark-compare=0001",
        # Absolute time comparison in seconds
        "--benchmark-compare-fail=mean:0.0005",
        *SHARED_BENCHMARK_FLAGS,
    )
