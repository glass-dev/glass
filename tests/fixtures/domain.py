"""Domain specific fixtures."""

from __future__ import annotations

import importlib.util
from typing import TYPE_CHECKING

import pytest

import glass

if TYPE_CHECKING:
    from collections.abc import Callable
    from types import ModuleType

    from glass._types import FloatArray
    from glass.cosmology import Cosmology


HAVE_NUMPY = importlib.util.find_spec("numpy") is not None


class MockCosmology:
    """A mock class to represent Cosmology in tests."""

    def __init__(self, xp: ModuleType) -> None:
        self.xp: ModuleType = xp

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
        z: FloatArray,
        z2: FloatArray | None = None,
    ) -> FloatArray:
        """Comoving distance :math:`d_c(z)` in Mpc."""
        return (
            self.xp.divide(self.xm(z), 1_000.0)
            if z2 is None
            else self.xp.divide(self.xm(z, z2), 1_000.0)
        )

    def inv_comoving_distance(self, dc: FloatArray) -> FloatArray:
        """Inverse function for the comoving distance in Mpc."""
        return 1_000 * (1 / (dc + self.xp.finfo(self.xp.float64).eps))

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


@pytest.fixture(scope="session")
def generate_cosmo() -> Callable[[ModuleType], Cosmology]:
    """Return a callable to generate a mock cosmology object."""
    return lambda xp: MockCosmology(xp=xp)


@pytest.fixture(scope="session")
def cosmo(xp: ModuleType) -> Cosmology:
    """Mock cosmology to use for core tests."""
    return MockCosmology(xp=xp)


@pytest.fixture(scope="session")
def cosmob(xpb: ModuleType) -> Cosmology:
    """Mock cosmology to use for benchmarking."""
    return MockCosmology(xp=xpb)


@pytest.fixture(scope="session")
def shells(xp: ModuleType) -> list[glass.RadialWindow]:  # noqa: D103
    return [
        glass.RadialWindow(
            np.asarray([0.0, 1.0, 2.0]),
            np.asarray([0.0, 1.0, 0.0]),
            1.0,
        ),
        glass.RadialWindow(
            np.asarray([1.0, 2.0, 3.0]),
            np.asarray([0.0, 1.0, 0.0]),
            2.0,
        ),
        glass.RadialWindow(
            np.asarray([2.0, 3.0, 4.0]),
            np.asarray([0.0, 1.0, 0.0]),
            3.0,
        ),
        glass.RadialWindow(
            np.asarray([3.0, 4.0, 5.0]),
            np.asarray([0.0, 1.0, 0.0]),
            4.0,
        ),
        glass.RadialWindow(
            np.asarray([4.0, 5.0, 6.0]),
            np.asarray([0.0, 1.0, 0.0]),
            5.0,
        ),
    ]
