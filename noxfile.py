"""Nox config."""

from __future__ import annotations

import nox

# Options to modify nox behaviour
nox.options.default_venv_backend = "uv|virtualenv"
nox.options.reuse_existing_virtualenvs = True
nox.options.sessions = ["lint", "tests"]

ALL_PYTHON = ["3.8", "3.9", "3.10", "3.11", "3.12"]


@nox.session
def lint(session: nox.Session) -> None:
    """Run the linter."""
    session.install("pre-commit")
    session.run("pre-commit", "run", "--all-files", *session.posargs)


@nox.session(python=ALL_PYTHON)
def tests(session: nox.Session) -> None:
    """Run the unit tests."""
    session.install("-c", ".github/test-constraints.txt", "-e", ".[test]")
    session.run(
        "pytest",
        *session.posargs,
    )


@nox.session(python=ALL_PYTHON)
def coverage(session: nox.Session) -> None:
    """Run tests and compute coverage."""
    session.posargs.append("--cov=glass")
    tests(session)


@nox.session(python=ALL_PYTHON)
def doctests(session: nox.Session) -> None:
    """Run the doctests."""
    session.posargs.append("--doctest-plus")
    session.posargs.append("glass")
    tests(session)


@nox.session
def examples(session: nox.Session) -> None:
    """Run the example notebooks."""
    session.install("-e", ".[examples]")
    session.run("jupyter", "execute", "examples/**/*.ipynb", *session.posargs)


@nox.session
def docs(session: nox.Session) -> None:
    """Build the docs. Pass "serve" to serve."""
    session.install("-e", ".[docs]")
    session.chdir("docs")
    session.run("sphinx-build", "-M", "html", ".", "_build")

    port = 8001

    if session.posargs:
        if "serve" in session.posargs:
            print(f"Launching docs at http://localhost:{port}/ - use Ctrl-C to quit")
            session.run("python", "-m", "http.server", f"{port}", "-d", "_build/html")
        else:
            print("Unsupported argument to docs")


@nox.session
def build(session: nox.Session) -> None:
    """Build an SDist and wheel."""
    session.install("build")
    session.run("python", "-m", "build")
