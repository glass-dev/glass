"""Module for cosmology.api utilities."""

from typing import TYPE_CHECKING, Any, Protocol

import cosmology.api

if TYPE_CHECKING:
    from numpy.typing import NDArray

    Array = NDArray[Any]


class Cosmology(
    cosmology.api.HasComovingDistance[Array, Array],
    cosmology.api.HasOmegaM0[Array],
    Protocol,
):
    """
    Cosmology protocol for GLASS.

    GLASS requires a cosmology object with the following methods
    and attributes, as defined by the Cosmology API:
    * :meth:`cosmology.api.H_over_H0`
    * ...

    """
