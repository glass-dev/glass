"""Module for cosmology.api utilities."""

from typing import Any, Protocol

from numpy.typing import NDArray

import cosmology.api


class Cosmology(
    cosmology.api.HasComovingDistance[NDArray[Any], NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasCriticalDensity0[NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasGrowthFactor[NDArray[Any], NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasHoverH0[NDArray[Any], NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasHubbleDistance[NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasInverseComovingDistance[NDArray[Any], NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasLittleH[NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasOmegaM0[NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasOmegaM[NDArray[Any], NDArray[Any]],  # type: ignore[misc]
    cosmology.api.HasTransverseComovingDistance[NDArray[Any], NDArray[Any]],  # type: ignore[misc]
    Protocol,
):
    """Cosmology protocol for GLASS."""
