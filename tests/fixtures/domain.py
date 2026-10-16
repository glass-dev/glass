"""Domain specific fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from glass._types import FloatArray


@pytest.fixture(scope="session")
def get_cosmo() -> Callable[..., MockCosmology]:
    """Create a MockCosmology instance for any backend."""
    return MockCosmology


@pytest.fixture(scope="session")
def cosmo(np: ModuleType) -> MockCosmology:
    """
    Mock cosmology to use for core tests.

    We make use of Numpy here instead of xp as this more closely
    emulates the potential setup of the user who may use a Cosmology
    library which is not array api compatible.
    """
    return MockCosmology(np)


@pytest.fixture(scope="session")
def cosmo_with_inverse_comoving_distance(
    np: ModuleType,
) -> MockCosmologyWithInverseComovingDistance:
    """
    Mock cosmology with inverse comoving distance to use for core tests.

    We make use of Numpy here instead of xp as this more closely
    emulates the potential setup of the user who may use a Cosmology
    library which is not array api compatible.
    """
    return MockCosmologyWithInverseComovingDistance(np)


@pytest.fixture(scope="session")
def get_cosmo_with_omega_m() -> Callable[..., MockCosmologyWithOmegaM]:
    """Create a MockCosmologyWithOmegaM instance for any backend."""
    return MockCosmologyWithOmegaM


@pytest.fixture(scope="session")
def cosmo_with_omega_m(np: ModuleType) -> MockCosmologyWithOmegaM:
    """
    Mock cosmology with OmegaM to use for core tests.

    We make use of Numpy here instead of xp as this more closely
    emulates the potential setup of the user who may use a Cosmology
    library which is not array api compatible.
    """
    return MockCosmologyWithOmegaM(np)


class MockCosmology:
    """A mock class to represent Cosmology in tests."""

    xp: ModuleType

    def __init__(self, xp: ModuleType) -> None:
        self.xp = xp

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
            return self.xp.asarray(z) * 1_000
        return (self.xp.asarray(z2) - self.xp.asarray(z)) * 1_000

    def rho_m_z(self, z: FloatArray) -> FloatArray:
        """Redshift-dependent matter density in Msol Mpc-3."""
        return self.critical_density0 * self.Omega_m0 * (1 + z) ** 3

    def comoving_distance(
        self,
        z: float | FloatArray,
        z2: float | FloatArray | None = None,
    ) -> FloatArray:
        """Comoving distance :math:`d_c(z)` in Mpc."""
        return self.xp.divide(self.xm(z, z2), 1_000.0)

    def transverse_comoving_distance(
        self,
        z: FloatArray,
        z2: FloatArray | None = None,
    ) -> FloatArray:
        """Transverse comoving distance :math:`d_M(z)` in Mpc."""
        return self.hubble_distance * self.xm(z, z2)


class MockCosmologyWithInverseComovingDistance(MockCosmology):
    """Mock cosmology with inverse comoving distance."""

    def inv_comoving_distance(self, dc: FloatArray) -> FloatArray:
        """Inverse function for the comoving distance in Mpc."""
        return 1_000 * (1 / (dc + self.xp.finfo(self.xp.float64).eps))


class MockCosmologyWithOmegaM(MockCosmology):
    """A mock class to represent Cosmology with OmegaM in tests."""

    def Omega_m(self, z: FloatArray) -> FloatArray:  # noqa: N802
        """Matter density parameter at redshift z."""
        return self.rho_m_z(z) / self.critical_density0


@pytest.fixture(scope="session")
def get_shells() -> Callable[..., list[glass.RadialWindow]]:  # noqa: D103
    return lambda xp: [
        glass.RadialWindow(
            xp.asarray([0.0, 1.0, 2.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            1.0,
        ),
        glass.RadialWindow(
            xp.asarray([1.0, 2.0, 3.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            2.0,
        ),
        glass.RadialWindow(
            xp.asarray([2.0, 3.0, 4.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            3.0,
        ),
        glass.RadialWindow(
            xp.asarray([3.0, 4.0, 5.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            4.0,
        ),
        glass.RadialWindow(
            xp.asarray([4.0, 5.0, 6.0]),
            xp.asarray([0.0, 1.0, 0.0]),
            5.0,
        ),
    ]


@pytest.fixture(scope="session")
def shells(  # noqa: D103
    get_shells: Callable[..., list[glass.RadialWindow]], xp: ModuleType
) -> list[glass.RadialWindow]:
    return get_shells(xp)


@pytest.fixture(scope="session")
def get_cl() -> Callable[..., FloatArray]:
    """Generate a consistent cl for a chosen array backend."""
    return lambda xp: 1e-2 / (2 * xp.arange(100 + 1) + 1) ** 2
