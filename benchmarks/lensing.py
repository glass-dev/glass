"""Benchmark for a realistic example lensing simulation."""

from __future__ import annotations

import os
from ast import literal_eval
from typing import TYPE_CHECKING

import jax
import numpy as np
from benchmark_utils import CosmologyWrapper, run_benchmark, xp_available_backends

# use the CAMB cosmology that generated the matter power spectra
import camb  # ty: ignore[unresolved-import]
from cosmology.compat.camb import Cosmology  # ty: ignore[unresolved-import]

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb  # ty: ignore[unresolved-import]
from glass import rng

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from glass._types import AngularPowerSpectra, FloatArray, UnifiedGenerator
    from glass.shells import RadialWindow


def lensing_benchmark(xp: ModuleType) -> None:
    """
    Realistic lensing benchmark.

    Includes setup steps and core simulation to be timed.
    """
    # cosmology for the simulation
    h = 0.7
    Oc = 0.25
    Ob = 0.05

    # basic parameters of the simulation
    nside = lmax = 128

    # set up CAMB parameters for matter angular power spectrum
    pars = camb.set_params(
        H0=100 * h,
        omch2=Oc * h**2,
        ombh2=Ob * h**2,
        NonLinear=camb.model.NonLinear_both,
    )
    results = camb.get_background(pars)

    # get the cosmology from CAMB
    cosmo = CosmologyWrapper(cosmo=Cosmology(results), cosmo_xp=np, xp=xp)

    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)

    # linear radial window functions
    shells = glass.linear_windows(zb)

    # linear radial window functions using numpy to allow calling camb
    shells_np = glass.linear_windows(np.asarray(zb))

    # compute the angular matter power spectra of the shells with CAMB
    cls = [xp.asarray(cl) for cl in glass.ext.camb.matter_cls(pars, lmax, shells_np)]  # ty:ignore[unresolved-attribute]

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)

    # set up lognormal fields for simulation
    fields = glass.lognormal_fields(shells)

    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    def timed_function(  # noqa: PLR0913
        *,
        cosmo: CosmologyWrapper,
        fields: Sequence[glass.grf.Lognormal],
        gls: AngularPowerSpectra,
        nside: int,
        shells: list[RadialWindow],
        xp: ModuleType,
    ) -> FloatArray:
        """Core simulation of the Realistic lensing benchmark to be timed."""
        urng: UnifiedGenerator = rng.default_rng(xp=xp)

        # this will compute the convergence field iteratively
        convergence = glass.MultiPlaneConvergence(cosmo)

        # generator for lognormal matter fields
        matter = glass.generate(fields, gls, nside, ncorr=3, rng=urng)

        # main loop to simulate the matter fields iterative
        for i, delta_i in enumerate(matter):
            # add lensing plane from the window function of this shell
            convergence.add_window(delta_i, shells[i])

            # compute shear field
            glass.from_convergence(convergence.kappa, shear=True)  # ty: ignore[no-matching-overload]

        return convergence.kappa  # ty: ignore[invalid-return-type]

    # Run benchmark passing convergence and matter
    run_benchmark(
        timed_function,
        cosmo=cosmo,
        fields=fields,
        gls=gls,
        nside=nside,
        shells=shells,
        xp=xp,
    )


RUN_PROFILE: bool = literal_eval(os.environ.get("RUN_PROFILE", "False"))

# Run benchmarks for each requested backend
for xp in xp_available_backends.values():
    if RUN_PROFILE and xp.__name__ == "jax.numpy":
        with jax.profiler.trace("jax_trace", create_perfetto_trace=True):
            lensing_benchmark(xp)
    else:
        lensing_benchmark(xp)
