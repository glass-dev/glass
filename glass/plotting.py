# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''optional module for plotting functions'''

__all__ = [
    'plot_cls',
]


import os
import logging

from itertools import product
from sortcl import cl_indices

from .typing import WorkDir, Cls, GaussianCls, RegGaussianCls, RegularizedCls


log = logging.getLogger('glass.plotting')


def plot_cls(workdir: WorkDir = None,
             cls: Cls = None,
             gaussian_cls: GaussianCls = None,
             reg_gaussian_cls: RegGaussianCls = None,
             regularized_cls: RegularizedCls = None) -> None:
    '''create triangle plots of cls'''

    import matplotlib.pyplot as plt

    if workdir is None:
        log.info('workdir not set, skipping...')
        return

    def _plot(filename, cls, name, other_cls=None, other_name=None):
        # number of fields from number of cls
        n = int((2*len(cls))**0.5)

        fig, ax = plt.subplots(n, n, figsize=(n, n))

        for i, j in product(range(n), range(n)):
            if i <= j:
                ax[i, j].tick_params(which='both', direction='in', labelsize=4, pad=1, size=3)
            if i > j:
                ax[i, j].axis('off')

        for _cls, _name, _c, _a in [(cls, name, 'k', 1.),
                                    (other_cls, other_name, 'r', 0.5)]:
            if _cls is not None:
                log.info('plotting %s...', _name)
                for i, j, cl in zip(*cl_indices(n), _cls):
                    ax[i, j].loglog(+cl, ls='-', c=_c, alpha=_a)
                    ax[i, j].loglog(-cl, ls='--', c=_c, alpha=_a)

        fig.tight_layout(pad=0.)

        log.info('writing %s...', filename)
        fig.savefig(os.path.join(workdir, filename), bbox_inches='tight')

    if cls is not None or regularized_cls is not None:
        _plot('cls.pdf', cls, 'cls', regularized_cls, 'regularized cls')

    if gaussian_cls is not None or reg_gaussian_cls is not None:
        _plot('gaussian_cls.pdf', gaussian_cls, 'Gaussian cls', reg_gaussian_cls, 'regularized Gaussian cls')
