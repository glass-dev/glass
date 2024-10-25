"""
Random points
=============

.. currentmodule:: glass

The following functions provide functionality for simulating point
processes on the sphere and sampling random positions.

Sampling
--------

.. autofunction:: positions_from_delta
.. autofunction:: uniform_positions
.. autofunction:: position_weights


Bias
----

.. autofunction:: effective_bias


Bias models
-----------

.. autofunction:: linear_bias
.. autofunction:: loglinear_bias

"""  # noqa: D205, D400

from __future__ import annotations

import typing

import healpix
import numpy as np
import numpy.typing as npt

from glass.core.array import broadcast_first, broadcast_leading_axes, trapz_product

if typing.TYPE_CHECKING:
    import collections.abc

    from glass.shells import RadialWindow


ARCMIN2_SPHERE = 60**6 // 100 / np.pi


def effective_bias(
    z: npt.NDArray[np.float64],
    bz: npt.NDArray[np.float64],
    w: RadialWindow,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    z
        _description_
    bz
        _description_
    w
        _description_

    Returns
    -------
        _description_

    """
    norm = np.trapz(  # type: ignore[attr-defined]
        w.wa,
        w.za,
    )
    return trapz_product((z, bz), (w.za, w.wa)) / norm


def linear_bias(
    delta: npt.NDArray[np.float64],
    b: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    delta
        _description_
    b
        _description_

    Returns
    -------
        _description_

    """
    return b * delta


def loglinear_bias(
    delta: npt.NDArray[np.float64],
    b: float | npt.NDArray[np.float64],
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    delta
        _description_
    b
        _description_

    Returns
    -------
        _description_

    """
    delta_g = np.log1p(delta)
    delta_g *= b
    np.expm1(delta_g, out=delta_g)
    return delta_g


def positions_from_delta(  # noqa: PLR0912, PLR0913, PLR0915
    ngal: float | npt.NDArray[np.float64],
    delta: npt.NDArray[np.float64],
    bias: float | npt.NDArray[np.float64] | None = None,
    vis: npt.NDArray[np.float64] | None = None,
    *,
    bias_model: str | typing.Callable[..., typing.Any] = "linear",
    remove_monopole: bool = False,
    batch: int = 1_000_000,
    rng: np.random.Generator | None = None,
) -> collections.abc.Generator[
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int | npt.NDArray[np.int_],
    ]
]:
    """
    _summary_.

    Parameters
    ----------
    PLR0913
        _description_
    PLR0915ngal
        _description_
    delta
        _description_
    bias
        _description_
    vis
        _description_
    bias_model
        _description_
    remove_monopole
        _description_
    batch
        _description_
    rng
        _description_

    Returns
    -------
        _description_

    Yields
    ------
        _description_

    Raises
    ------
    TypeError
        _description_

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # get the bias model
    if isinstance(bias_model, str):
        bias_model_callable = globals()[f"{bias_model}_bias"]
    elif not callable(bias_model):
        raise TypeError("bias_model must be string or callable")  # noqa: EM101,TRY003
    else:
        bias_model_callable = bias_model

    # broadcast inputs to common shape of extra dimensions
    inputs: list[tuple[float | npt.NDArray[np.float64], int]] = [(ngal, 0), (delta, 1)]
    if bias is not None:
        inputs.append((bias, 0))
    if vis is not None:
        inputs.append((vis, 1))
    dims, *rest = broadcast_leading_axes(*inputs)
    ngal, delta, *rest = rest
    if bias is not None:
        bias, *rest = rest
    if vis is not None:
        vis, *rest = rest

    # iterate the leading dimensions
    for k in np.ndindex(dims):
        # compute density contrast from bias model, or copy
        n = (
            np.copy(delta[k])
            if bias is None
            else bias_model_callable(delta[k], bias[k])
        )

        # remove monopole if asked to
        if remove_monopole:
            n -= np.mean(n, keepdims=True)

        # turn into number count, modifying the array in place
        n += 1
        n *= ARCMIN2_SPHERE / n.size * ngal[k]

        # apply visibility if given
        if vis is not None:
            n *= vis[k]

        # clip number density at zero
        np.clip(n, 0, None, out=n)

        # sample actual number in each pixel
        n = rng.poisson(n)

        # total number of points
        count = n.sum()
        # don't go through pixels if there are no points
        if count == 0:
            continue

        # for converting randomly sampled positions to HEALPix indices
        npix = n.shape[-1]
        nside = healpix.npix2nside(npix)

        # create a mask to report the count in the right axis
        cmask: int | npt.NDArray[np.int_]
        if dims:
            cmask = np.zeros(dims, dtype=int)
            cmask[k] = 1
        else:
            cmask = 1

        # sample the map in batches
        step = 1000
        start, stop, size = 0, 0, 0
        while count:
            # tally this group of pixels
            q = np.cumsum(n[stop : stop + step])
            # does this group of pixels fill the batch?
            if size + q[-1] < min(batch, count):
                # no, we need the next group of pixels to fill the batch
                stop += step
                size += q[-1]
            else:
                # how many pixels from this group do we need?
                stop += int(np.searchsorted(q, batch - size, side="right"))
                # if the first pixel alone is too much, use it anyway
                if stop == start:
                    stop += 1
                # sample this batch of pixels
                ipix = np.repeat(np.arange(start, stop), n[start:stop])
                lon, lat = healpix.randang(nside, ipix, lonlat=True, rng=rng)
                # next batch
                start, size = stop, 0
                # keep track of remaining number of points
                count -= ipix.size
                # yield the batch
                yield lon, lat, ipix.size * cmask

        # make sure that the correct number of pixels was sampled
        assert np.sum(n[stop:]) == 0  # noqa: S101


def uniform_positions(
    ngal: float | npt.NDArray[np.int_] | npt.NDArray[np.float64],
    *,
    rng: np.random.Generator | None = None,
) -> collections.abc.Generator[
    tuple[
        npt.NDArray[np.float64],
        npt.NDArray[np.float64],
        int | npt.NDArray[np.int_],
    ]
]:
    """
    _summary_.

    Parameters
    ----------
    ngal
        _description_
    rng
        _description_

    Returns
    -------
        _description_

    Yields
    ------
        _description_

    """
    # get default RNG if not given
    if rng is None:
        rng = np.random.default_rng()

    # sample number of galaxies
    ngal = rng.poisson(np.multiply(ARCMIN2_SPHERE, ngal))

    # extra dimensions of the output
    dims = np.shape(ngal)

    # make sure ntot is an array even if scalar
    ngal = np.broadcast_to(ngal, dims)

    # sample each set of points
    for k in np.ndindex(dims):
        # sample uniformly over the sphere
        lon = rng.uniform(-180, 180, size=ngal[k])
        lat = np.rad2deg(np.arcsin(rng.uniform(-1, 1, size=ngal[k])))

        # report count
        count: int | npt.NDArray[np.int_]
        if dims:
            count = np.zeros(dims, dtype=int)
            count[k] = ngal[k]
        else:
            count = int(ngal[k])

        yield lon, lat, count


def position_weights(
    densities: npt.NDArray[np.float64],
    bias: npt.NDArray[np.float64] | None = None,
) -> npt.NDArray[np.float64]:
    """
    _summary_.

    Parameters
    ----------
    densities
        _description_
    bias
        _description_

    Returns
    -------
        _description_

    """
    # bring densities and bias into the same shape
    if bias is not None:
        densities, bias = broadcast_first(densities, bias)
    # normalise densities after shape has been fixed
    densities = densities / np.sum(densities, axis=0)
    # apply bias after normalisation
    if bias is not None:
        densities = densities * bias
    # densities now contains the relative contribution with bias applied
    return densities
