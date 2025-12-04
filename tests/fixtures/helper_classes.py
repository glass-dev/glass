"""Helper classes with static methods to generate fixtures."""

from __future__ import annotations

import contextlib
from typing import TYPE_CHECKING

import numpy as np
import pytest

if TYPE_CHECKING:
    from collections.abc import Generator
    from types import ModuleType
    from typing import Any

    from glass._types import AnyArray, FloatArray, IntArray


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

    @staticmethod
    def assert_equal(actual: AnyArray, desired: AnyArray) -> None:
        """Check if two objects are not equal."""
        np.testing.assert_equal(actual, desired)


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
        lon = xp.empty(0)
        lat = xp.empty(0)
        cnt: IntArray = 0
        for lo, la, co in pos:
            lon = xp.concat([lon, lo])
            lat = xp.concat([lat, la])
            cnt = cnt + co
        return lon, lat, cnt


@pytest.fixture(scope="session")
def data_transformer() -> type[DataTransformer]:
    """Fixture for generator-consuming utility."""
    return DataTransformer


class GeneratorConsumer:
    """Helper class for fully consuming genertors in tests."""

    @staticmethod
    def consume(
        generator: Generator[Any],
        *,
        valid_exception: str = "No exception should have been thrown",
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
