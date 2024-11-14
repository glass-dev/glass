"""
Namespace loader for GLASS extension packages.

Uses the pkgutil namespace mechanism to find "ext" submodules of
packages that provide a "glass" module.

"""

import os.path
from pkgutil import extend_path


def _extend_path(path: list[str], name: str) -> list[str]:
    """
    Extend the path to include the "ext" submodules of packages.

    Parameters
    ----------
    path
        The path to extend.
    name
        The name of the package.

    Returns
    -------
        The extended path.

    """
    _pkg, _, _mod = name.partition(".")

    return list(
        filter(
            os.path.isdir,
            (os.path.join(p, _mod) for p in extend_path(path, _pkg)),  # noqa: PTH118
        ),
    )


__path__ = _extend_path(__path__, __name__)
