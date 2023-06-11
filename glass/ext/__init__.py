'''Namespace loader for GLASS extension packages.

Uses the pkgutil namespace mechanism to find "ext" submodules of
packages that provide a "glass" module.

'''


def _extend_path(path, name):
    import os.path
    from pkgutil import extend_path

    _pkg, _, _mod = name.partition('.')

    return list(filter(os.path.isdir,
                       (os.path.join(p, _mod)
                        for p in extend_path(path, _pkg))))


__path__ = _extend_path(__path__, __name__)
