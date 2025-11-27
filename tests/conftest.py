from __future__ import annotations

import contextlib
import importlib.metadata
import logging
import os
from typing import TYPE_CHECKING

import numpy as np
import packaging.version
import pytest

import glass
import glass._array_api_utils as _utils

with contextlib.suppress(ImportError):
    # only import if jax is available
    import glass.jax

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType
    from typing import Any

    from cosmology import Cosmology

    from glass._types import AnyArray, FloatArray, UnifiedGenerator


# Handling of array backends, inspired by-
# https://github.com/scipy/scipy/blob/36e349b6afbea057cb713fc314296f10d55194cc/scipy/conftest.py#L139

# environment variable to specify array backends for testing
# can be:
#   a particular array library (numpy, jax, array_api_strict, ...)
#   all (try finding every supported array library available in the environment)
ARRAY_BACKEND: str = os.environ.get("ARRAY_BACKEND", "")

# Change jax logger to only log ERROR or worse
logging.getLogger("jax").setLevel(logging.ERROR)


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
    _check_version("numpy", "2.1.0")
    xp_available_backends["numpy"] = np


def _import_and_add_array_api_strict(
    xp_available_backends: dict[str, ModuleType],
) -> None:
    """Add array_api_strict to the backends dictionary."""
    import array_api_strict

    _check_version("array_api_strict", "2.0.0")
    xp_available_backends["array_api_strict"] = array_api_strict
    array_api_strict.set_array_api_strict_flags(api_version="2024.12")


def _import_and_add_jax(xp_available_backends: dict[str, ModuleType]) -> None:
    """Add jax to the backends dictionary."""
    import jax

    _check_version("jax", "0.4.32")
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
    with contextlib.suppress(ImportError):
        _import_and_add_numpy(xp_available_backends)

    with contextlib.suppress(ImportError):
        _import_and_add_array_api_strict(xp_available_backends)

    with contextlib.suppress(ImportError):
        _import_and_add_jax(xp_available_backends)
else:
    msg = f"unsupported array backend: {ARRAY_BACKEND}"
    raise ValueError(msg)


# Pytest fixtures
@pytest.fixture(params=xp_available_backends.values(), scope="session")
def xp(request: pytest.FixtureRequest) -> ModuleType:
    """
    Fixture for array backend.

    Access array library functions using `xp.` in tests.
    """
    return request.param  # type: ignore[no-any-return]


@pytest.fixture(scope="session")
def uxpx(xp: ModuleType) -> _utils.XPAdditions:
    """
    Fixture for array backend.

    Access array library functions using `xp.` in tests.
    """
    return _utils.XPAdditions(xp)


@pytest.fixture(scope="session")
def urng(xp: ModuleType) -> UnifiedGenerator:
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
        return _utils.Generator(seed=seed)
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

        def H_over_H0(self, z: FloatArray) -> FloatArray:  # noqa: N802
            """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
            return (self.Omega_m0 * (1 + z) ** 3 + 1 - self.Omega_m0) ** 0.5

        def xm(
            self,
            z: FloatArray,
            z2: FloatArray | None = None,
        ) -> FloatArray:
            """
            Dimensionless transverse comoving distance.

            :math:`x_M(z) = d_M(z)/d_H`
            """
            if z2 is None:
                return np.array(z) * 1_000
            return (np.array(z2) - np.array(z)) * 1_000

        def rho_m_z(self, z: FloatArray) -> FloatArray:
            """Redshift-dependent matter density in Msol Mpc-3."""
            return self.critical_density0 * self.Omega_m0 * (1 + z) ** 3

        def comoving_distance(
            self,
            z: FloatArray,
            z2: FloatArray | None = None,
        ) -> FloatArray:
            """Comoving distance :math:`d_c(z)` in Mpc."""
            return self.xm(z) / 1_000 if z2 is None else self.xm(z, z2) / 1_000

        def inv_comoving_distance(self, dc: FloatArray) -> FloatArray:
            """Inverse function for the comoving distance in Mpc."""
            return 1_000 * (1 / (dc + np.finfo(float).eps))

        def Omega_m(self, z: FloatArray) -> FloatArray:  # noqa: N802
            """Matter density parameter at redshift z."""
            return self.rho_m_z(z) / self.critical_density0

        def transverse_comoving_distance(
            self,
            z: FloatArray,
            z2: FloatArray | None = None,
        ) -> FloatArray:
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


class Compare:
    """
    Helper class for array comparisons in tests.

    This class wraps numpy testing functions to provide a consistent interface
    for comparing arrays in tests. Ultimately, it would be great if we can
    make the array testing backend-agnostic.
    """

    @staticmethod
    def assert_allclose(
        actual: AnyArray,
        desired: AnyArray,
        *,
        rtol: float = 1e-7,
        atol: float = 0,
    ) -> None:
        """Check if two objects are not equal up to desired tolerance."""
        np.testing.assert_allclose(actual, desired, rtol=rtol, atol=atol)

    @staticmethod
    def assert_array_almost_equal_nulp(
        actual: AnyArray,
        desired: AnyArray,
        *,
        nulp: int = 1,
    ) -> None:
        """Compare two arrays relatively to their spacing."""
        np.testing.assert_array_almost_equal_nulp(actual, desired, nulp=nulp)

    @staticmethod
    def assert_array_equal(actual: AnyArray, desired: AnyArray) -> None:
        """Check if two array objects are not equal."""
        np.testing.assert_array_equal(actual, desired)

    @staticmethod
    def assert_array_less(actual: AnyArray, desired: AnyArray) -> None:
        """Check if two array objects are not ordered by less than."""
        np.testing.assert_array_less(actual, desired)

    @staticmethod
    def assert_equal(actual: AnyArray, desired: AnyArray) -> None:
        """Check if two objects are not equal."""
        np.testing.assert_equal(actual, desired)


@pytest.fixture(scope="session")
def compare() -> type[Compare]:
    """Fixture for array comparison utility."""
    return Compare


class GeneratorConsumer:
    """Helper class for fully consuming genertors in tests."""

    @staticmethod
    def consume(
        generator: Generator[Any],
    ) -> list[Any]:
        """
        Generate and consume a generator returned by a given functions.

        The resulting generator will be consumed an any ValueError
        exceptions swallowed.
        """
        output: list[Any] = []
        with contextlib.suppress(ValueError):
            # Consume in a loop, as we expect users to
            output.extend(iter(generator))
        return output


@pytest.fixture(scope="session")
def generator_consumer() -> type[GeneratorConsumer]:
    """Fixture for generator-consuming utility."""
    return GeneratorConsumer
