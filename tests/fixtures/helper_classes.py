"""Helper classes with static methods to generate fixtures."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType
    from typing import Any

    from glass._types import AnyArray, FloatArray, IntArray, UnifiedGenerator


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


@pytest.fixture(scope="session")
def compare() -> type[Compare]:
    """Fixture for array comparison utility."""
    return Compare


class DataTransformer:
    """Helper class for transforming various data structures into others."""

    @staticmethod
    def catpos(
        pos: Generator[
            tuple[
                FloatArray,
                FloatArray,
                IntArray,
            ]
        ],
        *,
        xp: ModuleType,
    ) -> tuple[
        FloatArray,
        FloatArray,
        IntArray,
    ]:
        """Concatenate an array of pos into three arrays lon, lat and count."""
        lons = []
        lats = []
        counts = []

        for lo, la, co in pos:
            lons.append(xp.asarray(lo))
            lats.append(xp.asarray(la))
            counts.append(xp.asarray(co))

        if lons:
            lon = xp.concat(lons, axis=0)
            lat = xp.concat(lats, axis=0)
            count = xp.sum(xp.stack(counts, axis=0), axis=0)
        else:
            lon = xp.empty(0)
            lat = xp.empty(0)
            count = xp.asarray(0)

        return lon, lat, count


@pytest.fixture(scope="session")
def data_transformer() -> type[DataTransformer]:
    """Fixture for generator-consuming utility."""
    return DataTransformer


class GeneratorConsumer:
    """Helper class for fully consuming generators in tests."""

    @staticmethod
    def consume(
        generator: Generator[Any],
        *,
        valid_exception: str = "",
    ) -> list[Any]:
        """
        Generate and consume a generator returned by a given functions.

        The resulting generator will be consumed an any ValueError
        exceptions swallowed.
        """
        output: list[Any] = []
        try:
            # Consume in a loop, as we expect users to
            output.extend(iter(generator))
        except ValueError as e:
            assert str(e) == valid_exception  # noqa: PT017
        return output


@pytest.fixture(scope="session")
def generator_consumer() -> type[GeneratorConsumer]:
    """Fixture for generator-consuming utility."""
    return GeneratorConsumer


class HealpixInputs:
    """Helper class for calculating inputs for HEALPix functions."""

    alm_size: int = 78
    lmax: int = 11
    npix: int = 192
    npts: int = 250
    nside: int = 4

    @staticmethod
    def alm(*, rng: UnifiedGenerator) -> FloatArray:
        """Generate random alm coefficients."""
        return rng.standard_normal(HealpixInputs.alm_size) + 1j * rng.standard_normal( # ty: ignore[unsupported-operator]
            HealpixInputs.alm_size
        )

    @staticmethod
    def fl(*, rng: UnifiedGenerator) -> FloatArray:
        """Generate random function of l."""
        return rng.standard_normal(HealpixInputs.lmax + 1)

    @staticmethod
    def ipix(*, rng: UnifiedGenerator, xp: ModuleType) -> IntArray:
        """Generate a list of HEALPix pixels."""
        cnts = rng.poisson(
            HealpixInputs.npts / HealpixInputs.npix,
            size=HealpixInputs.npix,
        )
        return xp.repeat(xp.arange(HealpixInputs.npix), cnts)

    @staticmethod
    def kappa(*, rng: UnifiedGenerator) -> FloatArray:
        """Generate a kappa map."""
        return rng.normal(size=HealpixInputs.npix)

    @staticmethod
    def latitudes(max_phi: float, *, rng: UnifiedGenerator) -> FloatArray:
        """Generate an array of latitudes."""
        return rng.uniform(-max_phi, max_phi, size=HealpixInputs.npts)

    @staticmethod
    def longitudes(max_theta: float, *, rng: UnifiedGenerator) -> FloatArray:
        """Generate an array of longitudes."""
        return rng.uniform(-max_theta, max_theta, size=HealpixInputs.npts)


@pytest.fixture(scope="session")
def healpix_inputs() -> type[HealpixInputs]:
    """Fixture for generating HEALPix inputs."""
    return HealpixInputs
