# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''command line runtime'''

import numpy as np
import logging
import argparse
import configparser
import sys

from importlib import import_module
from ast import literal_eval
from cosmology import LCDM

from . import __version__ as version
from .simulation import Simulation, Ref


DEFAULT_MODULES = [
    'glass.cls',
    'glass.matter',
    'glass.lensing',
]


LOG_LEVELS = ['debug', 'info', 'warning', 'error', 'critical']


def getboolean(config, name):
    b = config.get(name, False)
    return True if b is None else bool(b)


def parse_arg(arg, *, filename='<config>', refs=False):
    try:
        arg = literal_eval(arg)
    except ValueError:
        if refs:
            arg = Ref(name=arg)
    except SyntaxError as e:
        e.filename = filename
        raise e from None
    return arg


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='glass', description='generator for large scale structure')
    parser.add_argument('config', type=argparse.FileType('r'), help='configuration file')
    parser.add_argument('--quiet', '-q', action='store_true', help='silence console output')
    parser.add_argument('--logfile', help='log to file')
    parser.add_argument('--loglevel', choices=LOG_LEVELS, default='info', help='level for file logging')
    parser.add_argument('--version', action='version', version=f'%(prog)s {version}')

    args = parser.parse_args()

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
    log_stderr = logging.StreamHandler(sys.stderr)
    log_stderr.setLevel('WARNING')
    log.addHandler(log_stderr)

    # logging to file
    if args.logfile:
        log_file = logging.FileHandler(args.logfile, 'w')
        log_file.setFormatter(logging.Formatter('%(asctime)s: %(levelname)s: %(message)s'))
        log.addHandler(log_file)

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
                mod = import_module(module)
            except ModuleNotFoundError:
                log.critical('could not import "%s" module', module)
                sys.exit(1)
            mod_all = getattr(mod, '__all__', [])
            for name in mod_all:
                if name in namespace:
                    log.warning('import "%s" from module "%s" shadows import from module "%s"', name, module, namespace[name].__module__)
                namespace[name] = getattr(mod, name)

        log.info('imported %d definitions', len(namespace))
        log.debug('definitions: %s', ', '.join(namespace.keys()))

        log.info('## config')

        if 'config' not in config:
            raise KeyError('missing section: config')

        nside = int(config['config']['nside'])
        zbins = np.fromstring(config['config']['zbins'], sep=' ')
        cls = config['config'].get('cls', None)
        allow_missing_cls = getboolean(config['config'], 'allow_missing_cls')

        log.info('nside: %d', nside)
        log.info('zbins: %s', zbins)
        log.info('cls: %s', cls)
        log.info('allow missing cls: %s', allow_missing_cls)

        sim = Simulation(nside=nside, zbins=zbins, allow_missing_cls=allow_missing_cls)

        if 'cosmology' in config:
            log.info('## cosmology')

            kwargs = {}
            for par in config['cosmology']:
                kwargs[par] = parse_arg(config['cosmology'][par], filename=args.config.name)

            cosmo = LCDM(**kwargs)

            sim.set_cosmology(cosmo)

            log.info('cosmology: %s', cosmo)

        if cls is not None:
            func = f'cls_from_{cls}'
            if func not in namespace:
                raise NameError(f'cannot get cls from {cls}: unknown function "{func}"')
            func = namespace[func]
            kwargs = {}

            sect = f'cls:{cls}'
            if sect not in config:
                raise KeyError(f'missing section: {sect}')

            log.info('## %s', sect)

            for par in config[sect]:
                if par == 'fields':
                    kwargs[par] = dict((_.split('=', 1)*2)[:2] for _ in config[sect][par].split())
                else:
                    kwargs[par] = parse_arg(config[sect][par], filename=args.config.name)

            name, call = sim.set_cls(func, **kwargs)

            log.info('%s: %s', name, call)

        if 'simulation' in config:
            log.info('## simulation')

            for label, func in config['simulation'].items():
                if func is None:
                    name, func = None, label
                else:
                    name = label
                if func not in namespace:
                    raise NameError(f'simulation: {label}: unknown function "{func}"')
                _func, _args, _kwargs = namespace[func], [], {}
                if label in config:
                    for par, arg in config[label].items():
                        if arg is None:
                            par, arg = None, par
                        arg = parse_arg(arg, filename=args.config.name, refs=True)
                        if par is None:
                            _args.append(arg)
                        else:
                            _kwargs[par] = arg

                name, call = sim.add(name, _func, *_args, **_kwargs)

                log.info('%s: %s', name, call)

        log.info('# run')

        sim.run()

    except Exception as e:
        log.exception('uncaught exception', exc_info=e)
        sys.exit(1)
