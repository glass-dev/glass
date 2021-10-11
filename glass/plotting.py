# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''optional module for plotting functions'''

__all__ = [
    'plot_cls',
]


import os
import logging

from itertools import product, chain
from sortcl import cl_indices

from .typing import WorkDir


log = logging.getLogger('glass.plotting')


def plot_cls(workdir: WorkDir = None, **out_cls) -> None:
    '''create triangle plots of cls'''

    import matplotlib.pyplot as plt

    if workdir is None:
        log.info('workdir not set, skipping...')
        return

    cls = out_cls.get('cls', None)
    regularized_cls = out_cls.get('regularized_cls', None)
    gaussian_cls = out_cls.get('gaussian_cls', None)
    reg_gaussian_cls = out_cls.get('reg_gaussian_cls', None)

    def _plot(filename, cls, name, other_cls=None, other_name=None):
        # get all individual fields for which there are cls
        fields = []
        for f in chain.from_iterable(cls.keys()):
            if f not in fields:
                fields.append(f)
        if other_cls is not None:
            for f in chain.from_iterable(other_cls.keys()):
                if f not in fields:
                    fields.append(f)

        # number of distinct fields
        n = len(fields)

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
                for i, j in zip(*cl_indices(n)):
                    if (fields[i], fields[j]) in _cls:
                        cl = _cls[fields[i], fields[j]]
                    elif (fields[j], fields[i]) in _cls:
                        cl = _cls[fields[j], fields[i]]
                    else:
                        cl = None
                    if cl is not None:
                        ax[i, j].loglog(+cl, ls='-', c=_c, alpha=_a)
                        ax[i, j].loglog(-cl, ls='--', c=_c, alpha=_a)

        fig.tight_layout(pad=0.)

        log.info('writing %s...', filename)
        fig.savefig(os.path.join(workdir, filename), bbox_inches='tight')

    if cls is not None or regularized_cls is not None:
        _plot('cls.pdf', cls, 'cls', regularized_cls, 'regularized cls')

    if gaussian_cls is not None or reg_gaussian_cls is not None:
        _plot('gaussian_cls.pdf', gaussian_cls, 'Gaussian cls', reg_gaussian_cls, 'regularized Gaussian cls')
