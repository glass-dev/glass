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

    for name, cls in out_cls.items():
        log.info('plotting %s...', name)

        # get all individual fields for which there are cls
        fields = []
        for f in chain.from_iterable(cls.keys()):
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

        for i, j in zip(*cl_indices(n)):
            if (fields[i], fields[j]) in cls:
                cl = cls[fields[i], fields[j]]
            elif (fields[j], fields[i]) in cls:
                cl = cls[fields[j], fields[i]]
            else:
                cl = None
            if cl is not None:
                ax[i, j].loglog(+cl, ls='-', c='k')
                ax[i, j].loglog(-cl, ls='--', c='k')

        fig.tight_layout(pad=0.)

        for i in range(n):
            ax[0, i].set_title(fields[i], size=6)
            ax[i, -1].yaxis.set_label_position('right')
            ax[i, -1].set_ylabel(fields[i], size=6, rotation=270, va='bottom')

        filename = f'{name}.pdf'

        log.info('writing %s...', filename)

        fig.savefig(os.path.join(workdir, filename), bbox_inches='tight')
