# author: Nicolas Tessore <n.tessore@ucl.ac.uk>
# license: MIT
'''module for user functions'''

import logging
from contextlib import contextmanager


@contextmanager
def logger(level='info'):
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
