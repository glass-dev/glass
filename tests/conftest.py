from __future__ import annotations

import contextlib
import importlib.metadata
import os
from typing import TYPE_CHECKING

import numpy as np
import packaging.version
import pytest

import glass
import glass._array_api_utils
import glass.jax

if TYPE_CHECKING:
    import types

    from numpy.typing import NDArray

    from cosmology import Cosmology

    from glass._array_api_utils import UnifiedGenerator


# Handling of array backends, inspired by-
# https://github.com/scipy/scipy/blob/36e349b6afbea057cb713fc314296f10d55194cc/scipy/conftest.py#L139

# environment variable to specify array backends for testing
# can be:
#   a particular array library (numpy, jax, array_api_strict, ...)
#   all (try finding every supported array library available in the environment)
GLASS_ARRAY_BACKEND: str = os.environ.get("GLASS_ARRAY_BACKEND", "")


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
    array_api_strict.set_array_api_strict_flags(api_version="2024.12")


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


# Pytest fixtures
@pytest.fixture(params=xp_available_backends.values(), scope="session")
def xp(request: pytest.FixtureRequest) -> types.ModuleType:
    """
    Fixture for array backend.

    Access array library functions using `xp.` in tests.
    """
    return request.param


@pytest.fixture(scope="session")
def urng(xp: types.ModuleType) -> UnifiedGenerator:
    """
    Fixture for a unified RNG interface.

    Access the relevant RNG using `urng.` in tests.

    Must be used with the `xp` fixture. Use `rng` for non array API tests.
    """
    seed = 42
    backend = xp.__name__
    if backend == "jax.numpy":
        return glass.jax.Generator(seed=seed)
    if backend == "numpy":
        return np.random.default_rng(seed=seed)
    if backend == "array_api_strict":
        return glass._array_api_utils.Generator(seed=seed)
    msg = "the array backend in not supported"
    raise NotImplementedError(msg)


@pytest.fixture(scope="session")
def rng() -> np.random.Generator:
    """
    RNG fixture for non array API tests.

    Use `urng` for array API tests.
    """
    return np.random.default_rng(seed=42)


@pytest.fixture(scope="session")
def cosmo() -> Cosmology:
    class MockCosmology:
        @property
        def Omega_m0(self) -> float:  # noqa: N802
            """Matter density parameter at redshift 0."""
            return 0.3

        @property
        def critical_density0(self) -> float:
            """Critical density at redshift 0 in Msol Mpc-3."""
            return 3e4

        @property
        def hubble_distance(self) -> float:
            """Hubble distance in Mpc."""
            return 4.4e3

        def H_over_H0(self, z: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N802
            """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
            return (self.Omega_m0 * (1 + z) ** 3 + 1 - self.Omega_m0) ** 0.5

        def xm(
            self,
            z: NDArray[np.float64],
            z2: NDArray[np.float64] | None = None,
        ) -> NDArray[np.float64]:
            """
            Dimensionless transverse comoving distance.

            :math:`x_M(z) = d_M(z)/d_H`
            """
            if z2 is None:
                return np.array(z) * 1_000
            return (np.array(z2) - np.array(z)) * 1_000

        def rho_m_z(self, z: NDArray[np.float64]) -> NDArray[np.float64]:
            """Redshift-dependent matter density in Msol Mpc-3."""
            return self.critical_density0 * self.Omega_m0 * (1 + z) ** 3

        def comoving_distance(
            self,
            z: NDArray[np.float64],
            z2: NDArray[np.float64] | None = None,
        ) -> NDArray[np.float64]:
            """Comoving distance :math:`d_c(z)` in Mpc."""
            return self.xm(z) / 1_000 if z2 is None else self.xm(z, z2) / 1_000

        def inv_comoving_distance(self, dc: NDArray[np.float64]) -> NDArray[np.float64]:
            """Inverse function for the comoving distance in Mpc."""
            return 1_000 * (1 / (dc + np.finfo(float).eps))

        def Omega_m(self, z: NDArray[np.float64]) -> NDArray[np.float64]:  # noqa: N802
            """Matter density parameter at redshift z."""
            return self.rho_m_z(z) / self.critical_density0

        def transverse_comoving_distance(
            self,
            z: NDArray[np.float64],
            z2: NDArray[np.float64] | None = None,
        ) -> NDArray[np.float64]:
            """Transverse comoving distance :math:`d_M(z)` in Mpc."""
            return self.hubble_distance * self.xm(z, z2)

    return MockCosmology()


@pytest.fixture(scope="session")
def shells() -> list[glass.RadialWindow]:
    return [
        glass.RadialWindow(np.array([0.0, 1.0, 2.0]), np.array([0.0, 1.0, 0.0]), 1.0),
        glass.RadialWindow(np.array([1.0, 2.0, 3.0]), np.array([0.0, 1.0, 0.0]), 2.0),
        glass.RadialWindow(np.array([2.0, 3.0, 4.0]), np.array([0.0, 1.0, 0.0]), 3.0),
        glass.RadialWindow(np.array([3.0, 4.0, 5.0]), np.array([0.0, 1.0, 0.0]), 4.0),
        glass.RadialWindow(np.array([4.0, 5.0, 6.0]), np.array([0.0, 1.0, 0.0]), 5.0),
    ]
