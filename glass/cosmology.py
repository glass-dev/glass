"""Module for cosmology.api utilities."""

from typing import Protocol

import cosmology.api

from glass._types import AnyArray


class Cosmology(
    cosmology.api.HasComovingDistance[AnyArray, AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasCriticalDensity0[AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasGrowthFactor[AnyArray, AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasHoverH0[AnyArray, AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasHubbleDistance[AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasInverseComovingDistance[AnyArray, AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasLittleH[AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasOmegaM0[AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasOmegaM[AnyArray, AnyArray],  # ty: ignore[invalid-type-arguments]
    cosmology.api.HasTransverseComovingDistance[AnyArray, AnyArray],  # ty: ignore[invalid-type-arguments]
    Protocol,
):
    """Cosmology protocol for GLASS."""
