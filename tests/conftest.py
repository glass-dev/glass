import contextlib
import importlib.metadata
import os
import types

import numpy as np
import numpy.typing as npt
import packaging.version
import pytest

from cosmology import Cosmology

from glass import RadialWindow

# environment variable to specify array backends for testing
# can be:
#   a particular array library (numpy, jax, array_api_strict, ...)
#   all (try finding every supported array library available in the environment)
GLASS_ARRAY_BACKEND: str | bool = os.environ.get("GLASS_ARRAY_BACKEND", False)


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


def _import_and_add_numpy(xp_available_backends: dict[str, types.ModuleType]) -> None:
    """Add numpy to the backends dictionary."""
    _check_version("numpy", "2.1.0")
    xp_available_backends.update({"numpy": np})


def _import_and_add_array_api_strict(
    xp_available_backends: dict[str, types.ModuleType],
) -> None:
    """Add array_api_strict to the backends dictionary."""
    import array_api_strict

    _check_version("array_api_strict", "2.0.0")
    xp_available_backends.update({"array_api_strict": array_api_strict})
    array_api_strict.set_array_api_strict_flags(api_version="2023.12")


def _import_and_add_jax(xp_available_backends: dict[str, types.ModuleType]) -> None:
    """Add jax to the backends dictionary."""
    import jax

    _check_version("jax", "0.4.32")
    xp_available_backends.update({"jax.numpy": jax.numpy})
    # enable 64 bit numbers
    jax.config.update("jax_enable_x64", val=True)


# a dictionary with all array backends to test
xp_available_backends: dict[str, types.ModuleType] = {}

# if no backend passed, use numpy by default
if not GLASS_ARRAY_BACKEND or GLASS_ARRAY_BACKEND == "numpy":
    _import_and_add_numpy(xp_available_backends)
elif GLASS_ARRAY_BACKEND == "array_api_strict":
    _import_and_add_array_api_strict(xp_available_backends)
elif GLASS_ARRAY_BACKEND == "jax":
    _import_and_add_jax(xp_available_backends)
# if all, try importing every backend
elif GLASS_ARRAY_BACKEND == "all":
    with contextlib.suppress(ImportError):
        _import_and_add_numpy(xp_available_backends)

    with contextlib.suppress(ImportError):
        _import_and_add_array_api_strict(xp_available_backends)

    with contextlib.suppress(ImportError):
        _import_and_add_jax(xp_available_backends)
else:
    msg = f"unsupported array backend: {GLASS_ARRAY_BACKEND}"
    raise ValueError(msg)

# use this as a decorator for tests involving array API compatible functions
array_api_compatible = pytest.mark.parametrize("xp", xp_available_backends.values())


@pytest.fixture(scope="session")
def cosmo() -> Cosmology:
    class MockCosmology:
        @property
        def omega_m(self) -> float:
            return 0.3

        def ef(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            return (self.omega_m * (1 + z) ** 3 + 1 - self.omega_m) ** 0.5

        def xm(
            self,
            z: npt.NDArray[np.float64],
            z2: npt.NDArray[np.float64] | None = None,
        ) -> npt.NDArray[np.float64]:
            if z2 is None:
                return np.array(z) * 1000
            return (np.array(z2) - np.array(z)) * 1000

    return MockCosmology()


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def shells() -> list[RadialWindow]:
    return [
        RadialWindow(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]), 1.0),
        RadialWindow(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]), 2.0),
        RadialWindow(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0]), 3.0),
        RadialWindow(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 0.0]), 4.0),
        RadialWindow(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 0.0]), 5.0),
    ]
