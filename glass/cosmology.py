"""Module for cosmology.api utilities."""

from typing import Any, Protocol

import cosmology.api

from numpy.typing import NDArray

Array = NDArray[Any]


class Cosmology(
    cosmology.api.HasComovingDistance[Array, Array],  # type: ignore[misc]
    cosmology.api.HasCriticalDensity0[Array],  # type: ignore[misc]
    cosmology.api.HasHoverH0[Array, Array],  # type: ignore[misc]
    cosmology.api.HasHubbleDistance[Array],  # type: ignore[misc]
    cosmology.api.HasOmegaM0[Array],  # type: ignore[misc]
    cosmology.api.HasOmegaM[Array, Array],  # type: ignore[misc]
    cosmology.api.HasTransverseComovingDistance[Array, Array],  # type: ignore[misc]
    Protocol,
):
    """
    Cosmology protocol for GLASS.

    GLASS requires a cosmology object with the following methods
    and attributes, as defined by the Cosmology API:
    * :meth:`cosmology.api.H_over_H0`
    * ...

    """
