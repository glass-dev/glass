"""GLASS package."""

import contextlib

with contextlib.suppress(ModuleNotFoundError):
    from ._version import __version__, __version_tuple__  # noqa: F401
