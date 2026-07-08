"""Benchmark for a realistic example lensing simulation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from benchmark_utils import CosmologyWrapper, run_benchmark, xp_available_backends

# use the CAMB cosmology that generated the matter power spectra
import camb  # ty: ignore[unresolved-import]
from cosmology.compat.camb import Cosmology  # ty: ignore[unresolved-import]

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb  # ty: ignore[unresolved-import]
from glass import _rng

if TYPE_CHECKING:
    from types import ModuleType

    from glass._types import AnyArray, FloatArray, UnifiedGenerator
    from glass.shells import RadialWindow


# Run benchmarks for each requested backend
for xp in xp_available_backends.values():
    urng: UnifiedGenerator = _rng.rng_dispatcher(xp=xp)

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
    cls = [xp.asarray(cl) for cl in glass.ext.camb.matter_cls(pars, lmax, shells_np)]  # ty: ignore[unresolved-attribute]

    # apply discretisation to the full set of spectra:
    # - HEALPix pixel window function (`nside=nside`)
    # - maximum angular mode number (`lmax=lmax`)
    # - number of correlated shells (`ncorr=3`)
    cls = glass.discretized_cls(cls, nside=nside, lmax=lmax, ncorr=3)

    # set up lognormal fields for simulation
    fields = glass.lognormal_fields(shells)

    # compute Gaussian spectra for lognormal fields from discretised spectra
    gls = glass.solve_gaussian_spectra(fields, cls)

    # generator for lognormal matter fields
    matter = glass.generate(fields, gls, nside, ncorr=3, rng=urng)

    # this will compute the convergence field iteratively
    convergence = glass.MultiPlaneConvergence(cosmo)

    # localised redshift distribution
    # the actual density per arcmin2 does not matter here, it is never used
    z = xp.linspace(0.0, 1.0, 101)
    dndz = xp.exp(-((z - 0.5) ** 2) / (0.1) ** 2)

    # distribute dN/dz over the radial window functions
    ngal = glass.partition(z, dndz, shells)

    shape = 12 * nside**2

    def lensing_benchmark(  # noqa: PLR0913
        *,
        convergence: glass.MultiPlaneConvergence,
        matter: StopIteration[AnyArray],
        ngal: FloatArray,
        shape: tuple[int, ...],
        shells: list[RadialWindow],
        xp: ModuleType,
    ) -> tuple[FloatArray, FloatArray, FloatArray]:
        """Realistic lensing simulation benchmark."""
        kappa_bar = xp.zeros(shape)
        gamm1_bar = xp.zeros(shape)
        gamm2_bar = xp.zeros(shape)

        # main loop to simulate the matter fields iterative
        for i, delta_i in enumerate(matter):
            # add lensing plane from the window function of this shell
            convergence.add_window(delta_i, shells[i])

            # get convergence field
            kappa_i = convergence.kappa

            # compute shear field
            gamm1_i, gamm2_i = glass.from_convergence(kappa_i)

            # add to mean fields using the galaxy number density as weight
            kappa_bar += ngal[i] * kappa_i  # ty: ignore[unsupported-operator]
            gamm1_bar += ngal[i] * gamm1_i  # ty: ignore[unsupported-operator]
            gamm2_bar += ngal[i] * gamm2_i  # ty: ignore[unsupported-operator]

        # normalise mean fields by the total galaxy number density
        kappa_bar /= xp.sum(ngal)
        gamm1_bar /= xp.sum(ngal)
        gamm2_bar /= xp.sum(ngal)

    # Run benchmark passing convergence and matter
    run_benchmark(
        lensing_benchmark,
        convergence=convergence,
        matter=matter,
        ngal=ngal,
        shape=shape,
        shells=shells,
        xp=xp,
    )
