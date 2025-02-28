"""Module for cosmology.api utilities."""

from typing import Protocol

import cosmology.api


class Cosmology(
    cosmology.api.HasHoverH0,
    cosmology.api.HasOmegaM0,
    cosmology.api.HasHubbleDistance,
    Protocol,
):
    """
    Cosmology protocol for GLASS.

    GLASS requires a cosmology object with the following methods
    and attributes, as defined by the Cosmology API:
    * :meth:`cosmology.api.H_over_H0`
    * ...

    """
