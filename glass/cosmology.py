"""Module for cosmology.api utilities."""

from typing import TYPE_CHECKING, Protocol

import cosmology.api

if TYPE_CHECKING:
    from typing import Any

    from numpy.typing import NDArray

    Array = NDArray[Any]


class Cosmology(
    cosmology.api.HasComovingDistance[Array, Array],  # type: ignore[misc]
    cosmology.api.HasCriticalDensity0[Array],  # type: ignore[misc]
    cosmology.api.HasGrowthFactor[Array, Array],  # type: ignore[misc]
    cosmology.api.HasH0[Array],  # type: ignore[misc]
    cosmology.api.HasHoverH0[Array, Array],  # type: ignore[misc]
    cosmology.api.HasHubbleDistance[Array],  # type: ignore[misc]
    cosmology.api.HasInverseComovingDistance[Array, Array],  # type: ignore[misc]
    cosmology.api.HasOmegaM0[Array],  # type: ignore[misc]
    cosmology.api.HasOmegaM[Array, Array],  # type: ignore[misc]
    cosmology.api.HasTransverseComovingDistance[Array, Array],  # type: ignore[misc]
    Protocol,
):
    """Cosmology protocol for GLASS."""
