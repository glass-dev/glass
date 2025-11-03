"""Module for cosmology.api utilities."""

from typing import Protocol

import cosmology.api

from glass._types import AnyArray


class Cosmology(
    cosmology.api.HasComovingDistance[AnyArray, AnyArray],  # type: ignore[misc]
    cosmology.api.HasCriticalDensity0[AnyArray],  # type: ignore[misc]
    cosmology.api.HasGrowthFactor[AnyArray, AnyArray],  # type: ignore[misc]
    cosmology.api.HasHoverH0[AnyArray, AnyArray],  # type: ignore[misc]
    cosmology.api.HasHubbleDistance[AnyArray],  # type: ignore[misc]
    cosmology.api.HasInverseComovingDistance[AnyArray, AnyArray],  # type: ignore[misc]
    cosmology.api.HasLittleH[AnyArray],  # type: ignore[misc]
    cosmology.api.HasOmegaM0[AnyArray],  # type: ignore[misc]
    cosmology.api.HasOmegaM[AnyArray, AnyArray],  # type: ignore[misc]
    cosmology.api.HasTransverseComovingDistance[AnyArray, AnyArray],  # type: ignore[misc]
    Protocol,
):
    """Cosmology protocol for GLASS."""
