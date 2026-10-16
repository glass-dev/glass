"""Benchmark for a realistic example lensing simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from benchmark_utils import CosmologyWrapper, run_benchmark

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb
import glass.rng

if TYPE_CHECKING:
    from collections.abc import Sequence
    from types import ModuleType

    from glass._types import AngularPowerSpectra, FloatArray
    from glass.shells import RadialWindow


def benchmark(xp: ModuleType) -> None:
    """
    Realistic lensing simulation benchmark.

    Parameters
    ----------
    XP
        The array backend for which this benchmark should be executed.
    """
    # cosmology for the simulation
    h = 0.7
    Oc = 0.25
    Ob = 0.05

    # basic parameters of the simulation
    nside = lmax = 256

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
    cls = [
        xp.asarray(cl)
        for cl in glass.ext.camb.matter_cls(  # ty: ignore[unresolved-attribute]
            pars,
            lmax,
            shells_np,
        )
    ]

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)

    # set up lognormal fields for simulation
    fields = glass.lognormal_fields(shells)

    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    def key_lensing_simulation(
        *,
        cosmo: CosmologyWrapper,
        fields: Sequence[glass.grf.Lognormal],
        gls: AngularPowerSpectra,
        nside: int,
        shells: list[RadialWindow],
    ) -> FloatArray:
        """Time the Realistic lensing simulation benchmark."""
        urng = glass.rng.default_rng(xp=xp)

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
        key_lensing_simulation,
        cosmo=cosmo,
        fields=fields,
        gls=gls,
        nside=nside,
        shells=shells,
        xp=xp,
    )
