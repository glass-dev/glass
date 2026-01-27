"""
Handling of array backends.

Ispired by
https://github.com/scipy/scipy/blob/36e349b6afbea057cb713fc314296f10d55194cc/scipy/conftest.py#L139
"""

from __future__ import annotations

import importlib.metadata
import os
from typing import TYPE_CHECKING

import numpy as np
import packaging.version
import pytest

import glass._array_api_utils as _utils

if TYPE_CHECKING:
    from types import ModuleType

# environment variable to specify array backends for testing
# can be:
#   - a particular array library (numpy, jax, array_api_strict, ...)
#   - all (try finding every supported array library available in the environment)
ARRAY_BACKEND: str = os.environ.get("ARRAY_BACKEND", "")


def _check_version(lib: str, array_api_compliant_version: str) -> None:
    """
    Check if installed library's version is compliant with the array API standard.

    Parameters
    ----------
    lib
        name of the library.
    array_api_compliant_version
        version of the library compliant with the array API standard.

    Raises
    ------
    ImportError
        If the installed version is not compliant with the array API standard.

    """
    lib_version = packaging.version.Version(importlib.metadata.version(lib))
    if lib_version < packaging.version.Version(array_api_compliant_version):
        msg = f"{lib} must be >= {array_api_compliant_version}; found {lib_version}"
        raise ImportError(msg)


def _import_and_add_numpy(xp_available_backends: dict[str, ModuleType]) -> None:
    """Add numpy to the backends dictionary."""
    _check_version("numpy", "2.2.6")
    xp_available_backends["numpy"] = np


def _import_and_add_array_api_strict(
    xp_available_backends: dict[str, ModuleType],
) -> None:
    """Add array_api_strict to the backends dictionary."""
    import array_api_strict  # noqa: PLC0415

    _check_version("array_api_strict", "2.3.1")
    xp_available_backends["array_api_strict"] = array_api_strict
    array_api_strict.set_array_api_strict_flags(api_version="2024.12")


def _import_and_add_jax(xp_available_backends: dict[str, ModuleType]) -> None:
    """Add jax to the backends dictionary."""
    import jax  # noqa: PLC0415

    _check_version("jax", "0.6.2")
    xp_available_backends["jax.numpy"] = jax.numpy
    # enable 64 bit numbers
    jax.config.update("jax_enable_x64", val=True)


# a dictionary with all array backends to test
xp_available_backends: dict[str, ModuleType] = {}

# if no backend passed, use numpy by default
if not ARRAY_BACKEND or ARRAY_BACKEND == "numpy":
    _import_and_add_numpy(xp_available_backends)
elif ARRAY_BACKEND == "array_api_strict":
    _import_and_add_array_api_strict(xp_available_backends)
elif ARRAY_BACKEND == "jax":
    _import_and_add_jax(xp_available_backends)
# if all, try importing every backend
elif ARRAY_BACKEND == "all":
    _import_and_add_numpy(xp_available_backends)
    _import_and_add_array_api_strict(xp_available_backends)
    _import_and_add_jax(xp_available_backends)
else:
    msg = f"unsupported array backend: {ARRAY_BACKEND}"
    raise ValueError(msg)


@pytest.fixture(params=xp_available_backends.values(), scope="session")
def xp(request: pytest.FixtureRequest) -> ModuleType:
    """
    Fixture for array backend.

    Access array library functions using `xp.` in tests.

    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(
    params=[
        xp
        for name, xp in xp_available_backends.items()
        if name not in {"array_api_strict", "jax.numpy"}
    ],
    scope="session",
)
def xpb(request: pytest.FixtureRequest) -> ModuleType:
    """
    Fixture for array backend to be used in benchmarks.

    Access array library functions using `xpb.` in tests.

    We are excluding array-api-strict and jax for two reasons
    1. Our use of array-api-strict is not for its performance but
       for checking our interface with array libraries. Additionally,
       users are unlikely to use array-api-strict with glass.
       Therefore, it is not worth benchmarking with array-api-strict.
    2. We did not previously support jax, therefore it does
       not _yet_ make sense to regression test jax as there is
       nothing to compare against, since jax is not supported by
       the older versions of glass.

    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def ap() -> ModuleType:
    """Fixture for the array_api_strict array backend."""
    return xp_available_backends["array_api_strict"]


@pytest.fixture(scope="session")
def jnp() -> ModuleType:
    """Fixture for the jax.numpy array backend."""
    return xp_available_backends["jax.numpy"]


@pytest.fixture(scope="session")
def uxpx(xp: ModuleType) -> _utils.XPAdditions:
    """
    Fixture for array backend.

    Access array library functions using `xp.` in tests.

    """
    return _utils.XPAdditions(xp)
