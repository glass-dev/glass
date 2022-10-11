# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for running GLASS scripts in a user-friendly wrapper'''

import sys
import argparse
import logging
from contextlib import contextmanager
from runpy import run_path
from pdb import post_mortem

from glass.core import GeneratorError


@contextmanager
def set_logger(level='info'):
    '''context manager for logging'''
    level = str(level).upper()
    log = logging.getLogger('glass')
    handler = logging.StreamHandler()
    formatter = logging.Formatter('%(message)s')
    handler.setFormatter(formatter)
    handler.setLevel(level)
    old_level = log.level
    log.addHandler(handler)
    log.setLevel(level)
    try:
        yield log
    finally:
        log.setLevel(old_level)
        log.removeHandler(handler)
        handler.close()


@contextmanager
def _update_argv(path, argv=[]):
    '''helper context manager to run script with modified argv'''
    sys_argv = sys.argv
    try:
        sys.argv = [path, *argv]
        yield
    finally:
        sys.argv = sys_argv


if __name__ == '__main__':
    parser = argparse.ArgumentParser(prog='python -m glass',
                                     description='run a GLASS script')
    parser.add_argument('-l', '--loglevel', default='info', help='control the level of logging')
    parser.add_argument('-D', '--debugger', action='store_true', help='enter debugger on error')
    parser.add_argument('path', help='path to Python script')
    parser.add_argument('args', nargs=argparse.REMAINDER, help='arguments for Python script')

    args = parser.parse_args()

    thescript = ' '.join([args.path, *args.args])

    with set_logger(args.loglevel) as logger:
        logger.info('running script: %s', thescript)
        try:
            with _update_argv(args.path, args.args):
                run_path(args.path, run_name='__main__')
        except BaseException as e:
            logger.exception('An uncaught exception occurred while running script: %s\n\n', thescript)
            if args.debugger:
                if isinstance(e, GeneratorError):
                    globals().update({
                        '__glass_generator__': e.generator,
                        '__glass_state__': e.state,
                    })
                    tb = e.__cause__.__traceback__
                else:
                    tb = e.__traceback__
                print('\nEntering post-mortem debugger ...\n')
                post_mortem(tb)
                print('\nExiting ...\n')
            sys.exit(1)
