# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for plotting and visualisation'''

import logging
import healpy as hp

from ._generator import generator


log = logging.getLogger('glass.plot')


def _nice_grid(n):
    '''return rows and columns for a nice grid of n plots'''
    from math import sqrt, ceil
    b = int(ceil(sqrt(n)) + 0.1)
    a = int(ceil(n/b) + 0.1)
    while True:
        if a*(b-1) >= n:
            b -= 1
        elif (a-1)*b >= n:
            a -= 1
        else:
            break
    return a, b


@generator('zmin, zmax, maps, points')
def interactive_display(map_names=[], point_names=[]):
    import matplotlib.pyplot as plt

    plots = [(name, 'map') for name in map_names]
    plots += [(name, 'points') for name in point_names]

    n = len(plots)
    nr, nc = _nice_grid(n)

    plt.ion()

    plt.figure()

    while True:
        try:
            zmin, zmax, maps, points = yield
        except GeneratorExit:
            break

        data = maps + points

        plt.suptitle(f'z = {zmin:.3f} ... {zmax:.3f}')

        for i in range(n):
            name, kind = plots[i]
            plt.subplot(nr, nc, i+1)
            if kind == 'map':
                hp.mollview(data[i], title=name, hold=True)
            elif kind == 'points':
                pass

        plt.pause(1e-3)

    plt.ioff()
