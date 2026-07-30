"""Helper functions for running glass benchmarks."""

from __future__ import annotations

import os
from timeit import timeit
from typing import TYPE_CHECKING

import jax
import numpy as np

import array_api_strict

if TYPE_CHECKING:
    from types import FunctionType, ModuleType
    from typing import Any

    from glass._types import FloatArray
    from glass.cosmology import Cosmology

xp_available_backends: dict[str, ModuleType] = {}

# environment variable to specify array backends for testing
# can be:
#   - a particular array library (numpy, jax, array_api_strict, ...)
#   - all (try finding every supported array library available in the environment)
ARRAY_BACKEND: str = os.environ.get("ARRAY_BACKEND", "")

# if no backend passed, use numpy by default
if not ARRAY_BACKEND or ARRAY_BACKEND == "numpy":
    xp_available_backends["numpy"] = np
elif ARRAY_BACKEND == "array_api_strict":
    xp_available_backends["array_api_strict"] = array_api_strict
elif ARRAY_BACKEND == "jax":
    xp_available_backends["jax.numpy"] = jax.numpy
# if all, try importing every backend
elif ARRAY_BACKEND == "all":
    xp_available_backends["numpy"] = np
    xp_available_backends["array_api_strict"] = array_api_strict
    xp_available_backends["jax.numpy"] = jax.numpy
else:
    msg = f"unsupported array backend: {ARRAY_BACKEND}"
    raise ValueError(msg)

print("Running benchmarks for backends: ", ", ".join(xp_available_backends.keys()))  # noqa: T201

# Configure backends
array_api_strict.set_array_api_strict_flags(api_version="2025.12")
jax.config.update("jax_enable_x64", val=True)


def run_benchmark(
    function_to_benchmark: FunctionType,
    *args: tuple[Any, ...],
    xp: ModuleType,
    **kwargs: dict[str, Any],
) -> None:
    """
    Run a benchmark of the provided function and its arguments.

    Parameters
    ----------
    function_to_benchmark
        The function which should be benchmarked. Note that the function must accept
        all values passed via `args`, `xp` and `kwargs`
    args
        Positional arguments to be passed to `function_to_benchmark`
    xp
        The array backend to benchmark with. Will also be passed as a named argument to
        `function_to_benchmark`
    kwargs
        Extra named arguments to be passed to `function_to_benchmark`
    """
    # benchmark the task
    result = timeit(lambda: function_to_benchmark(*args, xp=xp, **kwargs), number=1)
    # report the result
    print(f"Took {result:.3f} seconds with {xp.__name__}")  # noqa: T201


class CosmologyWrapper:
    """An Python Array API compatible wrapper for a Cosmology instance."""

    cosmo: Cosmology
    cosmo_xp: ModuleType
    xp: ModuleType

    def __init__(
        self, *, cosmo: Cosmology, cosmo_xp: ModuleType = np, xp: ModuleType
    ) -> None:
        self.cosmo = cosmo
        self.cosmo_xp = cosmo_xp
        self.xp = xp

    @property
    def Omega_m0(self) -> FloatArray:  # noqa: N802
        """Matter density parameter at redshift 0."""
        return self.xp.asarray(self.cosmo.Omega_m0)

    @property
    def critical_density0(self) -> FloatArray:
        """Critical density at redshift 0 in Msol Mpc-3."""
        return self.xp.asarray(self.cosmo.critical_density0)

    @property
    def hubble_distance(self) -> FloatArray:
        """Hubble distance in Mpc."""
        return self.xp.asarray(self.cosmo.hubble_distance)

    def H_over_H0(self, z: FloatArray) -> FloatArray:  # noqa: N802
        """Standardised Hubble function :math:`E(z) = H(z)/H_0`."""
        return self.xp.asarray(self.cosmo.H_over_H0(self.cosmo_xp.asarray(z)))

    def xm(
        self,
        z: FloatArray,
        z2: FloatArray | None = None,
    ) -> FloatArray:
        """
        Dimensionless transverse comoving distance.

        :math:`x_M(z) = d_M(z)/d_H`
        """
        if z2 is not None:
            z2 = self.cosmo_xp.asarray(z2)
        return self.xp.asarray(
            self.cosmo.xm(
                self.cosmo_xp.asarray(z),
                z2,
            )
        )

    def rho_m_z(self, z: FloatArray) -> FloatArray:
        """Redshift-dependent matter density in Msol Mpc-3."""
        return self.xp.asarray(self.cosmo.rho_m_z(self.cosmo_xp.asarray(z)))

    def comoving_distance(
        self,
        z: float,
        z2: float | None = None,
    ) -> FloatArray:
        """Comoving distance :math:`d_c(z)` in Mpc."""
        return self.xp.asarray(self.cosmo.comoving_distance(z, z2))

    def inv_comoving_distance(self, dc: FloatArray) -> FloatArray:
        """Inverse function for the comoving distance in Mpc."""
        return self.xp.asarray(
            self.cosmo.inv_comoving_distance(self.cosmo_xp.asarray(dc))
        )

    def Omega_m(self, z: FloatArray) -> FloatArray:  # noqa: N802
        """Matter density parameter at redshift z."""
        return self.xp.asarray(self.cosmo.Omega_m(self.cosmo_xp(z)))

    def transverse_comoving_distance(
        self,
        z: float,
        z2: float | None = None,
    ) -> FloatArray:
        """Transverse comoving distance :math:`d_M(z)` in Mpc."""
        return self.xp.asarray(self.cosmo.transverse_comoving_distance(z, z2))
