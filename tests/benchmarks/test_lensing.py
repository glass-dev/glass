"""Benchmarks for lensing example simulations."""

from typing import TYPE_CHECKING

# use the CAMB cosmology that generated the matter power spectra
import camb
from cosmology.compat.camb import Cosmology

# almost all GLASS functionality is available from the `glass` namespace
import glass
import glass.ext.camb

if TYPE_CHECKING:
    from types import ModuleType

    from pytest_benchmark.fixture import BenchmarkFixture


def test_lensing(benchmark: BenchmarkFixture, xp: ModuleType, urng):
    """Benchmark for a realistic example lensing simulation."""
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
    cosmo = Cosmology(results)

    # shells of 200 Mpc in comoving distance spacing
    zb = glass.distance_grid(cosmo, 0.0, 1.0, dx=200.0)

    # linear radial window functions
    shells = glass.linear_windows(zb)

    # compute the angular matter power spectra of the shells with CAMB
    cls = glass.ext.camb.matter_cls(pars, lmax, shells)

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

    def function_to_benchmark():
        # the integrated convergence and shear field over the redshift distribution
        kappa_bar = xp.zeros(12 * nside**2)
        gamm1_bar = xp.zeros(12 * nside**2)
        gamm2_bar = xp.zeros(12 * nside**2)

        # main loop to simulate the matter fields iterative
        for i, delta_i in enumerate(matter):
            # add lensing plane from the window function of this shell
            convergence.add_window(delta_i, shells[i])

            # get convergence field
            kappa_i = convergence.kappa

            # compute shear field
            gamm1_i, gamm2_i = glass.shear_from_convergence(kappa_i)

            # add to mean fields using the galaxy number density as weight
            kappa_bar += ngal[i] * kappa_i
            gamm1_bar += ngal[i] * gamm1_i
            gamm2_bar += ngal[i] * gamm2_i

        # normalise mean fields by the total galaxy number density
        kappa_bar /= ngal.sum()
        gamm1_bar /= ngal.sum()
        gamm2_bar /= ngal.sum()

    benchmark(function_to_benchmark)
