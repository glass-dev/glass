# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''command line runtime'''

import numpy as np
import logging
import argparse
import configparser
import sys

from importlib import import_module
from importlib.util import find_spec, module_from_spec, LazyLoader
from ast import literal_eval

from . import __version__ as version
from .simulation import Simulation
from .typing import annotate


DEFAULT_MODULES = [
    'glass.cosmology',
    'glass.cls',
    'glass.matter',
    'glass.lensing',
    'glass.analysis',
    'glass.plotting',
    'glass.observations',
    'glass.galaxies',
]


LOG_LEVELS = ['debug', 'info', 'warning', 'error', 'critical']


def lazy_import_module(name):
    spec = find_spec(name)
    if spec is None:
        return import_module(name)
    module = module_from_spec(spec)
    loader = LazyLoader(module.__loader__)
    loader.exec_module(module)
    return module


def getboolean(config, name):
    b = config.get(name, False)
    return True if b is None else config.getboolean(name, False)


def parse_arg(arg, sim, *, filename='<config>'):
    # only try and parse strings
    if not isinstance(arg, str):
        return None

    # nested configs
    if arg[0] == '\n':
        lines = arg[1:].split('\n')
        nested = {}
        for key, val, *_ in ((*map(str.strip, line.split('=', 1)), None) for line in lines if line):
            nested[key] = parse_arg(val or key, sim, filename=filename)
        return nested

    # try to literally parse the string
    try:
        arg = literal_eval(arg)
    except ValueError:
        literal = False
    except SyntaxError as e:
        e.filename = filename
        raise e from None
    else:
        literal = True

    # if not a literal, it must be a reference
    if not literal:
        arg = sim.ref(arg)

    return arg


def main(*args):
    parser = argparse.ArgumentParser(prog='glass', description='generator for large scale structure')
    parser.add_argument('config', type=argparse.FileType('r'), help='configuration file')
    parser.add_argument('--workdir', '-d', help='working directory for file output')
    parser.add_argument('--quiet', '-q', action='store_true', help='silence console output')
    parser.add_argument('--logfile', help='log to file')
    parser.add_argument('--loglevel', choices=LOG_LEVELS, default='info', help='level for file logging')
    parser.add_argument('--version', action='version', version=f'%(prog)s {version}')

    args = parser.parse_args(*args)

    log = logging.getLogger('glass')

    # the lowest log level the logger will deal with
    log.setLevel(args.loglevel.upper())

    # logging to console
    # info messages go to stdout, unless quiet
    # warnings and above go to stderr
    if not args.quiet:
        log_stdout = logging.StreamHandler(sys.stdout)
        log_stdout.setLevel('INFO')
        log_stdout.addFilter(lambda record: record.levelno == logging.INFO)
        log.addHandler(log_stdout)
    else:
        log_stdout = None
    log_stderr = logging.StreamHandler(sys.stderr)
    log_stderr.setLevel('WARNING')
    log.addHandler(log_stderr)

    # logging to file
    if args.logfile:
        log_file = logging.FileHandler(args.logfile, 'w')
        log_file.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        log.addHandler(log_file)
    else:
        log_file = None

    try:
        log.info('GLASS %s', version)

        log.info('# config')
        log.info('file: %s', args.config.name)

        config = configparser.ConfigParser(allow_no_value=True)
        config.optionxform = lambda option: option
        config.read_file(args.config)

        modules = [*DEFAULT_MODULES]

        log.info('## system')

        if 'system' in config:
            if 'modules' in config['system']:
                modules += config['system']['modules'].split()

        log.info('modules: %s', ', '.join(modules))

        log.info('importing modules...')
        namespace = {}
        for module in modules:
            try:
                mod = lazy_import_module(module)
            except ModuleNotFoundError as e:
                log.critical('could not import "%s" module: %s', module, e)
                sys.exit(1)
            mod_all = getattr(mod, '__glass__', [])
            for name in mod_all:
                if name in namespace:
                    log.warning('import "%s" from module "%s" shadows import from module "%s"', name, module, namespace[name].__module__)
                namespace[name] = getattr(mod, name)

        log.info('imported %d definitions', len(namespace))
        log.debug('definitions: %s', ', '.join(namespace.keys()))

        log.info('## config')

        if 'config' not in config:
            raise KeyError('missing section: config')

        workdir = args.workdir
        nside = int(config['config']['nside'])
        lmax = config['config'].getint('lmax')
        zbins = np.fromstring(config['config']['zbins'], sep=' ')
        allow_missing_cls = getboolean(config['config'], 'allow_missing_cls')

        log.info('workdir: %s', workdir)
        log.info('nside: %d', nside)
        log.info('lmax: %s', lmax)
        log.info('zbins: %s', zbins)
        log.info('allow missing cls: %s', allow_missing_cls)

        sim = Simulation(workdir=workdir, nside=nside, lmax=lmax, zbins=zbins, allow_missing_cls=allow_missing_cls)

        if 'simulation' in config:
            log.info('## simulation')

            for label, func in config['simulation'].items():
                if func is None:
                    name, func = None, label
                else:
                    name = label
                if func not in namespace:
                    raise NameError(f'simulation: {label}: unknown function "{func}"')
                _func, _kwargs = namespace[func], {}
                if label in config:
                    for par, arg in config[label].items():
                        _kwargs[par] = parse_arg(arg or par, sim, filename=args.config.name)

                if name is not None:
                    _func = annotate(_func, name)

                call = sim.add(_func, **_kwargs)

                log.info('%s', call)

        sim.run()

        # clean up log handlers
        for h in log_stdout, log_stderr, log_file:
            if h is not None:
                log.removeHandler(h)

    except Exception as e:
        log.exception('uncaught exception', exc_info=e)
        sys.exit(1)


if __name__ == '__main__':
    main()
