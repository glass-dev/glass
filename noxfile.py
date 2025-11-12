"""Nox config."""

import os
from pathlib import Path

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
GLASS_REPO_URL = "https://github.com/glass-dev/glass"


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

    session.run("pytest", *session.posargs)


@nox.session(python=ALL_PYTHON)
def coverage(session: nox.Session) -> None:
    """Run tests and compute coverage."""
    session.install(
        "-c",
        ".github/test-constraints.txt",
        "-e",
        ".",
        "--group",
        "coverage",
    )

    session.posargs.append("--cov")
    session.run("pytest", *session.posargs)


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

    session.posargs.append("--doctest-plus")
    session.posargs.append("--doctest-plus-generate-diff=overwrite")
    session.posargs.append("glass")
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
            *Path().glob("examples/**/*.ipynb"),
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
def benchmark(session: nox.Session) -> None:
    """Run the benchmarks."""
    if not session.posargs:
        msg = "Revision not provided"
        raise ValueError(msg)

    if len(session.posargs) == 1:
        revision = session.posargs[0]
    else:
        msg = (
            f"Incorrect number of revisions provided ({len(session.posargs)}), "
            f"expected 2"
        )
        raise ValueError(msg)

    session.install(f"git+{GLASS_REPO_URL}@{revision}", "pytest", "pytest-benchmark")

    session.posargs.append("--benchmark-autosave")
    session.run("pytest", *session.posargs)
