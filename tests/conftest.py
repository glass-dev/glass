import numpy as np
import numpy.typing as npt
import pytest

from cosmology import Cosmology

from glass import RadialWindow


@pytest.fixture(scope="session")
def cosmo() -> Cosmology:
    class MockCosmology:
        @property
        def omega_m(self) -> float:
            """Matter density parameter at redshift 0."""
            return 0.3

        @property
        def rho_c(self) -> float:
            """Critical density at redshift 0 in Msol Mpc-3."""
            return 3e4

        def ef(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
            return (self.omega_m * (1 + z) ** 3 + 1 - self.omega_m) ** 0.5

        def xm(
            self,
            z: npt.NDArray[np.float64],
            z2: npt.NDArray[np.float64] | None = None,
        ) -> npt.NDArray[np.float64]:
            """
            Dimensionless transverse comoving distance.

            :math:`x_M(z) = d_M(z)/d_H`
            """
            if z2 is None:
                return np.array(z) * 1_000
            return (np.array(z2) - np.array(z)) * 1_000

        def rho_m_z(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Redshift-dependent matter density in Msol Mpc-3."""
            return self.rho_c * self.omega_m * (1 + z) ** 3

        def dc(
            self,
            z: npt.NDArray[np.float64],
            z2: npt.NDArray[np.float64] | None = None,
        ) -> npt.NDArray[np.float64]:
            """Comoving distance :math:`d_c(z)` in Mpc."""
            return self.xm(z) / 1_000 if z2 is None else self.xm(z, z2) / 1_000

        def dc_inv(self, dc: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:
            """Inverse function for the comoving distance in Mpc."""
            return 1_000 * (1 / dc)

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
