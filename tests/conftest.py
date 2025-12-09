"""Main pytest configuration file."""

import logging

# Change jax logger to only log ERROR or worse
logging.getLogger("jax").setLevel(logging.ERROR)

pytest_plugins = [
    "tests.fixtures.array_backends",
    "tests.fixtures.domain",
    "tests.fixtures.generators",
    "tests.fixtures.helper_classes",
]
