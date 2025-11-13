"""Nox config."""

import os
import pathlib
import shutil

import nox

# Options to modify nox behaviour
nox.options.default_venv_backend = "uv|virtualenv"
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
]
ARRAY_BACKENDS = {
    "array_api_strict": "array-api-strict>=2",
    "jax": "jax>=0.4.32",
}
BENCH_TESTS_LOC = pathlib.Path("tests/benchmarks")
GLASS_REPO_URL = "https://github.com/glass-dev/glass"


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


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=ALL_PYTHON)
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.install(
        "-c",
        ".github/test-constraints.txt",
        "-e",
        ".",
        "--group",
        "test",
    )

    array_backend = os.environ.get("ARRAY_BACKEND")
    if array_backend == "array_api_strict":
        session.install(ARRAY_BACKENDS["array_api_strict"])
    elif array_backend == "jax":
        session.install(ARRAY_BACKENDS["jax"])
    elif array_backend == "all":
        session.install(*ARRAY_BACKENDS.values())

    session.run("pytest", *session.posargs, env=os.environ)


@nox.session(python=ALL_PYTHON)
def coverage(session: nox.Session) -> None:
    """Run tests and compute coverage for the core tests."""
    session.posargs.append("--cov")
    tests(session)


@nox.session(python=ALL_PYTHON)
def coverage_benchmarks(session: nox.Session) -> None:
    """Run tests and compute coverage for the benchmark tests."""
    session.posargs.extend([BENCH_TESTS_LOC, "--cov"])
    tests(session)


@nox.session(python=ALL_PYTHON)
def doctests(session: nox.Session) -> None:
    """Run the doctests."""
    session.install(
        "-c",
        ".github/test-constraints.txt",
        "-e",
        ".",
        "--group",
        "doctest",
    )

    session.posargs.extend(
        [
            "--doctest-plus",
            "--doctest-plus-generate-diff=overwrite",
            "glass",
        ]
    )
    session.run("pytest", *session.posargs)


@nox.session
def examples(session: nox.Session) -> None:
    """Run the example notebooks. Pass "html" to build html."""
    session.install("-e", ".[examples]")

    if session.posargs:
        if "html" in session.posargs:
            print("Generating HTML for the example notebooks")
            session.run(
                "jupyter",
                "nbconvert",
                "--to",
                "html",
                "--embed-images",
                "examples/**/*.ipynb",
            )
        else:
            print("Unsupported argument to examples")
    else:
        session.run(
            "jupyter",
            "execute",
            "--inplace",
            *pathlib.Path().glob("examples/**/*.ipynb"),
            *session.posargs,
        )


@nox.session
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.install("-e", ".", "--group", "docs")
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

            print(f"Launching docs at http://localhost:{port}/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", f"{port}", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.install("build")
    session.run("python", "-m", "build")


@nox.session
def version(session: nox.Session) -> None:
    """
    Check the current version of the package.

    The intent of this check is to ensure that the package
    is installed without any additional dependencies
    through optional dependencies nor dependency groups.
    """
    session.install("-e", ".")
    session.run("python", "-c", "import glass; print(glass.__version__)")


@nox.session(python=ALL_PYTHON)
def benchmarks(session: nox.Session) -> None:
    """
    Run the benchmark test for a specific revision.

    Note it is not possible to pass extra options to pytest.
    """
    _check_revision_count(session.posargs, expected_count=1)
    revision = session.posargs[0]

    # essentially required just for the dependencies
    session.install("-e", ".", "--group", "test")

    # overwrite current package with specified revision
    session.install(f"git+{GLASS_REPO_URL}@{revision}")
    session.run("pytest", BENCH_TESTS_LOC)


@nox.session(python=ALL_PYTHON)
def regression_tests(session: nox.Session) -> None:
    """
    Run regression benchmark tests between two revisions.

    Note it is not possible to pass extra options to pytest.
    """
    _check_revision_count(session.posargs, expected_count=2)
    before_revision, after_revision = session.posargs

    # essentially required just for the dependencies
    session.install("-e", ".", "--group", "test")

    # make sure benchmark directory is clean
    benchmark_dir = pathlib.Path(".benchmarks")
    if benchmark_dir.exists():
        session.log(f"Deleting previous benchmark directory: {benchmark_dir}")
        shutil.rmtree(benchmark_dir)

    print(f"Generating prior benchmark from revision {before_revision}")
    session.install(f"git+{GLASS_REPO_URL}@{before_revision}")
    session.run("pytest", BENCH_TESTS_LOC, "--benchmark-autosave")

    print(f"Comparing {before_revision} benchmark to revision {after_revision}")
    session.install(f"git+{GLASS_REPO_URL}@{after_revision}")
    session.run(
        "pytest",
        BENCH_TESTS_LOC,
        "--benchmark-compare=0001",
        "--benchmark-compare-fail=min:5%",
    )
