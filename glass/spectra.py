import itertools

import numpy as np
import pylevin as levin
from scipy.interpolate import RectBivariateSpline, UnivariateSpline
from tqdm import tqdm


def make_pk_interpolators(pk, k, z) -> RectBivariateSpline:
    """
    Spline interpolator for the matter power spectrum.

    Parameters
    ----------
    pk
        3D matter power spectrum.
    k
        Array of wavenumbers corresponding to 3D matter power spectrum values.
    z
        Array of redshifts corresponding to 3D matter power spectrum values.

    Returns
    -------
        RectBivariateSpline object to interpolate the 3D matter power spectrum.

    """
    pkz_interp = RectBivariateSpline(k, z, pk)
    return pkz_interp


def make_shell_interpolator(shell_z, shell_weights, cosmo) -> list[UnivariateSpline]:
    """
    Spline interpolator for comoving shells.

    Parameters
    ----------
    shell_z
        The redshift boundaries for concentric matter shells.
    shell_weights
        The weights of the shells.
    cosmo
        Cosmology object to calculate the comoving distances.

    Returns
    -------
        A list the length of the number of shells with each element being a
        UnivariateSpline for that shell.

    """
    chi = cosmo.comoving_distance(shell_z)
    dzdchi = UnivariateSpline(chi, shell_z, k=2, s=0).derivative()(chi)
    shell_interp = [
        UnivariateSpline(chi, shell * dzdchi, k=2, s=0) for shell in shell_weights
    ]
    return shell_interp


def make_glass_shell_interpolator(ws, cosmo) -> list[UnivariateSpline]:
    """
    Spline interpolator for comoving shells.

    Parameters
    ----------
    ws
        Glass shells.
    cosmo
        Cosmology object to calculate the comoving distances.

    Returns
    -------
        A list the length of the number of shells with each element being a
        UnivariateSpline for that shell.

    """
    return [
        UnivariateSpline(
            cosmo.comoving_distance(ws[i].za) * (1 + ws[i].za),
            ws[i].wa,
            k=2,
            s=0,
            ext=1,
        )
        for i in range(len(ws))
    ]


def get_cls(
    num_shells,
    ell_min,  # can be set to 1
    ell_max,
    pk_interpolator,
    shells_interpolate,
    chi_lims,
    # number of threads used for hyperthreading
    N_thread=4,
    # number of collocation points in each bisection
    n_sub=8,
    # maximum number of bisections used
    n_bisec_max=64,
    # relative accuracy target
    rel_acc=1e-10,
    # should the bessel functions be calculated with boost instead of GSL,
    # higher accuracy at high Bessel orders
    boost_bessel=True,
    # should the code talk to you?
    verbose=False,
    lev_list=None,
):
    """
    Calculates Cls using the geometric approximation and performing the integral
    in comoving distance first.

    Parameters
    ----------
    num_shells
        The number of shells.
    ell_min
        The minimum ell value.
    ell_max
        The maximum ell value.
    pk_interpolator
        The interpolator for the matter power spectrum.
    shells_interpolate
        The interpolator for the comoving shells.
    chi_lims
        The comoving distance limits for the shells.
    N_thread
        The number of threads used for hyperthreading.
    n_sub
        The number of collocation points in each bisection.
    n_bisec_max
        The maximum number of bisections used.
    rel_acc
        The relative accuracy target.
    boost_bessel
        Should the bessel functions be calculated with boost instead of GSL,
        higher accuracy at high Bessel orders.
    verbose
        Should the code talk to you?
    lev_list
        List of Levin objects to load. If None, new Levin objects are created.
        Useful for speeding up the calculation if we have at hand already
        calculated bisections.

    Returns
    -------
    shell_cls
        A 3D array of the shell cls.
    shell_cls_dict
        A dictionary with the shell cls labelled by the shell pair number.
    lev_list
        A list of Levin objects. If lev_list is None, this will be a list of new
        Levin objects created during the calculation. If lev_list is not None,
        this will be the same list passed in as lev_list.
    """
    # Some levin definitions
    integral_type = 0
    logx = True  # Tells the code to create a logarithmic spline in x for f(x)
    logy = True  # Tells the code to create a logarithmic spline in y for y = f(x)

    ell = np.unique((np.geomspace(ell_min, ell_max)).astype(int))

    if lev_list is None:
        lev_list = []
        load_levin = False
    else:
        lev_list
        load_levin = True

    N_int = int(2e3)
    k_int = np.geomspace(1e-4, 10, N_int)
    inner_int = np.zeros((num_shells, len(ell), len(k_int)))
    flat_idx = 0
    for idx_shell in tqdm(np.arange(num_shells)):
        chi_int = np.linspace(chi_lims[idx_shell], chi_lims[idx_shell + 1], 50)
        pk_nl_new = np.zeros((len(chi_int), len(k_int)))
        pk_nl_new[:, :] = (
            np.sqrt(pk_interpolator(k_int, chi_int)).T
            * (shells_interpolate[idx_shell](chi_int))[:, None]
        )
        lower_limit = chi_int[0] * np.ones_like(k_int)
        upper_limit = chi_int[-1] * np.ones_like(k_int)

        for i_ell, val_ell in enumerate(ell):
            if load_levin is True:
                lev = lev_list[flat_idx]
            else:
                lev = levin.pylevin(
                    integral_type, chi_int, pk_nl_new, logx, logy, N_thread, True
                )
                lev.set_levin(n_sub, n_bisec_max, rel_acc, boost_bessel, verbose)

            ell_values = (val_ell * np.ones_like(k_int)).astype(int)
            lev.levin_integrate_bessel_single(
                lower_limit,
                upper_limit,
                k_int,
                ell_values,
                inner_int[idx_shell, i_ell, :],
            )
            flat_idx += 1

            if load_levin is False:
                lev_list.append(lev)

    shell_cls = (
        2
        / np.pi
        * np.trapezoid(
            inner_int[:, None, :, :] * inner_int[None, :, :, :] * k_int**2,
            k_int,
            axis=-1,
        )
    )

    shell_cls_dict = {
        f"W{i + 1}xW{j + 1}": shell_cls[i, j, :]
        for i, j in itertools.product(
            range(len(chi_lims) - 1), range(len(chi_lims) - 1)
        )
    }
    return shell_cls, shell_cls_dict, lev_list
