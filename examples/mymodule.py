# this is an example of a custom module, see importing-modules.cfg

# this defines what functions the config can see
__all__ = [
    'my_plain_function',
    'my_other_plain_function',
    'my_print_function',
]


# imports are invisible to the config, unless added to __all__
import numpy as np


def my_plain_function(my_array_size):
    '''this is a plain function

    This function takes array size n and returns a random n-by-n array.

    '''

    return np.random.randn(my_array_size, my_array_size)


def my_other_plain_function(my_input_parameter):
    '''this is another plain function

    This function takes one argument and returns its negative.

    '''

    return -my_input_parameter


def my_print_function(my_name, my_array, my_other_name, my_other_array):
    '''this function prints its arguments

    This function prints its arguments. It does not return anything.

    '''

    print(my_name)
    print(my_array)
    print(my_other_name)
    print(my_other_array)
