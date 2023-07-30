
How *GLASS* works
=================

.. note::
   This page gives a fairly high-level overview of how *GLASS* works.  See the
   :doc:`list of GLASS publications <publications>` for additional in-depth
   references.


The main purpose of *GLASS* is to produce cosmological simulations on the
sphere.  The full, three-dimensional past light cone of the observer is
discretised into a sequence of nested shells, which are further discretised in
the angular dimensions into maps of the sphere.


Radial discretisation
---------------------

The discretisation in the radial (line of sight) direction is done in *GLASS*
using the concept of a :term:`radial window`, which consists of a window
function :math:`W` that assigns a weight :math:`W(z)` to each redshift
:math:`z`.  In the *GLASS* code, the :class:`~glass.shells.RadialWindow` named
tuple is used to define radial windows.

A sequence :math:`W_1, W_2, \ldots` of such window functions defines the shells
of the simulation.  For example, the :func:`~glass.shells.tophat_windows`
function takes redshift boundaries and returns a sequence of top hat windows,
which are flat and non-overlapping.

.. plot::

    from glass.shells import redshift_grid, tophat_windows

    # create a redshift grid for shell edges
    zs = redshift_grid(0., 0.5, dz=0.1)

    # create the top hat windows
    ws = tophat_windows(zs)

    # plot each window
    for i, (za, wa, zeff) in enumerate(ws):
        plt.plot(za, wa, c='k', lw=2)
        plt.fill_between(za, np.zeros_like(wa), wa, alpha=0.5)
        plt.annotate(f'shell {i+1}', (zeff, 0.5), ha='center', va='center')

    plt.xlabel('redshift $z$')
    plt.ylabel('window function $W(z)$')
    plt.tight_layout()

Given such a sequence of window functions :math:`W_i`, *GLASS* discretises a
continuous field :math:`F` (e.g. the matter density in the universe) by using
each :math:`W_i` in turn to project :math:`F` onto the sphere,

.. math::

    F_i = \frac{\int W_i(z) \, F(z) \, dz}{\int W_i(z) \, dz} \;.

This results in the sequence :math:`F_1, F_2, \ldots` of integrated (projected)
fields, which are :term:`spherical functions<spherical function>`.  *GLASS*
then simulates the (radially) continuous field :math:`F(z)` as the (radially)
discretised fields :math:`F_i`.


.. _user-window-functions:

Window functions
^^^^^^^^^^^^^^^^

*GLASS* supports arbitrary window functions (although the computation of
:ref:`line-of-sight integrals <user-los-integrals>` makes some assumptions).
The following :ref:`window functions <reference-window-functions>` are
included:

.. plot::

    from glass.shells import (redshift_grid, tophat_windows, linear_windows,
                              cubic_windows)

    plot_windows = [tophat_windows, linear_windows,
                    cubic_windows]
    nr = (len(plot_windows)+1)//2

    fig, axes = plt.subplots(nr, 2, figsize=(8, nr*3), layout="constrained",
                             squeeze=False, sharex=False, sharey=True)

    zs = redshift_grid(0., 0.5, dz=0.1)
    zt = np.linspace(0., 0.5, 200)

    for ax in axes.flat:
        ax.axis(False)
    for windows, ax in zip(plot_windows, axes.flat):
        ws = windows(zs)
        wt = np.zeros_like(zt)
        ax.axis(True)
        ax.set_title(windows.__name__)
        for i, (za, wa, zeff) in enumerate(ws):
            wt += np.interp(zt, za, wa, left=0., right=0.)
            ax.fill_between(za, np.zeros_like(wa), wa, alpha=0.5)
        ax.plot(zt, wt, c="k", lw=2)
    for ax in axes.flat:
        ax.set_xlabel("redshift $z$")
    for ax in axes[:, 0]:
        ax.set_ylabel("window function $W(z)$")


Angular discretisation
----------------------

The projected fields :math:`F_i` are still continuous functions on the sphere.
They therefore require further discretisation, which turns :math:`F_i` into a
spherical map of finite resolution.  In *GLASS*, this is done using the
*HEALPix* [#healpix]_ discretisation of the sphere.

Any spherical map is a discrete collection of spherical pixels :math:`F_{i,k}`,
:math:`k = 1, 2, \ldots`.  There are two ways that values can be assigned to
pixels:

1. Each pixel is set to the average of the field over its area, or
2. Each pixel is set to the function value at its centre.

In the first case, the discretised map :math:`F_{i,k}` is approximately a
convolution of the continuous projected field :math:`F_i` with a pixel kernel,
usually called the :term:`pixel window function`.  This convolution is then
sampled at the pixel centres.  In the second case, the continuous projected
field :math:`F_i` itself is sampled at the pixel centres.

*GLASS* can simulate either kind of angular discretisation.  The only
difference between the two is whether or not the pixel window function is
applied to the spherical harmonic expansion of the fields.


.. _user-los-integrals:

Line-of-sight integrals
-----------------------

The `radial discretisation`_ determines how well the simulation can approximate
line-of-sight integrals of the form

.. math::

    I(z) = \int_{0}^{z} \! a(z') \, F(z') \, dz' \;,

with :math:`a` some redshift-dependent factor, and :math:`F` a continuous field
simulated by *GLASS*.  Integrals of this kind appear e.g. when simulating
gravitational lensing or the distribution of galaxies.

To approximate such integrals using the discretised fields :math:`F_i`, three
additional requirements are imposed on the radial windows of the simulated
shells:

1. Every window has an associated effective redshift :math:`z_{\rm eff}` which
   is, in some sense, representative of the window. For example, this could be
   the mean or central redshift of the window function.
2. The window functions of shells :math:`j < i` vanish above the effective
   redshift :math:`z_{{\rm eff}, i}` of shell :math:`i`,

   .. math::

      W_j(z) = 0 \quad \text{if $j < i$ and $z \ge z_{{\rm eff}, i}$.}

3. The window functions of shells :math:`j > i` vanish below the effective
   redshift :math:`z_{{\rm eff}, i}` of shell :math:`i`,

   .. math::

      W_j(z) = 0 \quad \text{if $j > i$ and $z \le z_{{\rm eff}, i}$.}

In short, the requirements say that each shell has an effective redshift which
partitions the window functions of all other shells. In *GLASS*, it is stored
as the ``zeff`` attribute of :class:`~glass.shells.RadialWindow`.  Functions
that construct a list of windows for shells should ensure these requirements
are met.

To approximate the integral :math:`I(z)` using the projected fields
:math:`F_i`, it is evaluated in the effective redshifts of the windows as
:math:`I_i = I(z_{{\rm eff}, i})`.  Inserting the partition of unity

.. math::

   1 = \frac{\sum_{j} W_j(z)}{\sum_{j'} W_{j'}(z)}

into the integrand, and exchanging summation and integration,

.. math::

   I_i
   = \sum_{j \le i} \int_{0}^{z_{{\rm eff}, i}} \!
            a(z') \, \frac{W_j(z')}{\sum_{j'} W_{j'}(z')} \, F(z') \, dz' \;,

where the outer sum was truncated at :math:`j = i` using requirement 3.
Conversely, the remaining integrals can for :math:`j < i` be extended to
infinity using requirement 2.

Now the crucial part:  If the radial discretisation is sufficiently fine,
everything in the integrands except for :math:`W_j(z) \, F(z)` can be
approximated by its value in the effective redshift :math:`z_{{\rm eff}, j}`,

.. math::

   I_i
   \approx \sum_{j < i} a(z_{{\rm eff}, j}) \,
                    \frac{1}{W_j(z_{{\rm eff}, j})} \,
                    \int W_j(z') \, F(z') \, dz'
   + R_i \;,

where :math:`\sum_{j'} W_{j'}(z_{{\rm eff}, j}) = W_j(z_{{\rm eff}, j})` by
requirements 2 and 3 above, and :math:`R_i` is the remaining contribution of
shell :math:`i` to the integral,

.. math::

    R_i
    = \int_{0}^{z_{{\rm eff}, i}} \!
            a(z') \, \frac{W_i(z')}{\sum_{j'} W_{j'}(z')} \, F(z') \, dz' \;.

Overall, the approximation of the integral by the projected fields :math:`F_i`
is

.. math::

   I_i
   \approx \sum_{j < i} a(z_{{\rm eff}, j}) \,
        \frac{\int W_j(z) \, dz}{W_j(z_{{\rm eff}, j})} \, F_j
   + R_i \;.

It depends on the application whether :math:`R_i` is best approximated as zero,
or

.. math::

    R_i
    \approx a(z_{{\rm eff}, i}) \,
        \frac{\int W_i(z) \, dz}{W_i(z_{{\rm eff}, i})} \, F_i \;,

or set to some other value.

.. [#healpix] Gorski et al., 2005, ApJ, 622, 759,
   https://healpix.sourceforge.io
