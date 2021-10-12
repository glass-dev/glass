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


def plot_cls(workdir: WorkDir = None, /, **files) -> None:
    '''create triangle plots of cls'''

    import matplotlib.pyplot as plt

    if workdir is None:
        log.info('workdir not set, skipping...')
        return

    for file, file_cls in files.items():
        # get all individual fields for which there are cls
        fields = []
        for cls in file_cls.values():
            for f in chain.from_iterable(cls.keys()):
                if f not in fields:
                    fields.append(f)

        # number of distinct fields
        n = len(fields)

        # alpha value of lines
        _a = 2/(1+len(file_cls))

        fig, ax = plt.subplots(n, n, figsize=(n, n))

        for i, j in product(range(n), range(n)):
            if i <= j:
                ax[i, j].tick_params(which='both', direction='in', labelsize=4, pad=1, size=3)
            if i > j:
                ax[i, j].axis('off')

        log.info('plotting %s...', file)

        for name, cls in file_cls.items():
            log.info('- %s', name)

            for i, j in zip(*cl_indices(n)):
                if (fields[i], fields[j]) in cls:
                    cl = cls[fields[i], fields[j]]
                elif (fields[j], fields[i]) in cls:
                    cl = cls[fields[j], fields[i]]
                else:
                    cl = None
                if cl is not None:
                    _c = next(ax[i, j]._get_lines.prop_cycler)['color']
                    ax[i, j].loglog(+cl, ls='-', c=_c, label=name, alpha=_a)
                    ax[i, j].loglog(-cl, ls='--', c=_c, alpha=_a)

        fig.tight_layout(pad=0.)

        for i in range(n):
            ax[0, i].set_title(fields[i], size=6)
            ax[i, -1].yaxis.set_label_position('right')
            ax[i, -1].set_ylabel(fields[i], size=6, rotation=270, va='bottom')

        ax[0, 0].legend(loc='upper center', bbox_to_anchor=(0.5, -0.1), fontsize=6)

        filename = f'{file}.pdf'

        log.info('writing %s...', filename)

        fig.savefig(os.path.join(workdir, filename), bbox_inches='tight')
