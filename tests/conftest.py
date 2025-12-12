"""Main pytest configuration file."""

import logging

# Change jax logger to only log ERROR or worse
logging.getLogger("jax").setLevel(logging.ERROR)

pytest_plugins = [
    "fixtures.array_backends",
    "fixtures.domain",
    "fixtures.generators",
    "fixtures.helper_classes",
]
