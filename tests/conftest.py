import numpy as np
import numpy.typing as npt
import pytest

from cosmology import StandardCosmology

from glass import RadialWindow


@pytest.fixture
def cosmo() -> StandardCosmology[npt.NDArray[np.float64], npt.NDArray[np.float64]]:
    class MockCosmology:
        @property
        def Omega_m0(self) -> float:  # noqa: N802
            return 0.3

        def H_over_H0(self, z: npt.NDArray[np.float64]) -> npt.NDArray[np.float64]:  # noqa: N802
            return (self.Omega_m0 * (1 + z) ** 3 + 1 - self.Omega_m0) ** 0.5

        def xm(
            self,
            z: npt.NDArray[np.float64],
            z2: npt.NDArray[np.float64] | None = None,
        ) -> npt.NDArray[np.float64]:
            """Dimensionless transverse comoving distance."""
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
