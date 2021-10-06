# this is an example of a custom module, see custom-modules.cfg and
# custom-modules-with-annotations.cfg in that order

# this defines what functions the config can see
__all__ = [
    # for custom-modules.cfg
    'my_plain_function',
    'my_other_plain_function',
    'my_print_function',

    # for custom-modules-with-annotations.cfg
    'my_annotated_function',
    'my_other_annotated_function',
    'my_annotated_print_function',
]


# imports are invisible to the config, unless added to __all__
import numpy as np


####
# first a couple of plain python functions
# you could write these without ever having heard of GLASS
###


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


def my_print_function(my_array, my_other_array, my_name, my_other_name):
    '''this function prints its arguments

    This function prints its arguments. It does not return anything.

    '''

    print(my_name)
    print(my_array)
    print(my_other_name)
    print(my_other_array)


####
# now the magic
# these functions use GLASS-specific python type annotations
# this allows the functions to configure themselves
###


# some types already used throughout the code
from glass.types import Annotated, ArrayLike

# the Annotated type used by GLASS is simply imported from typing
# equivalent if you develop something GLASS-compatible but independent:
#   from typing import Annotated


def my_annotated_function(my_array_size=4) -> Annotated[ArrayLike, 'name:foo']:
    '''this is an annotated function

    Same as `my_plain_function()` but knows that the default name for its
    return value is "foo".  This is encoded as an annotation ``'name:foo'``
    in the return type.

    Also has a default argument now, so it doesn't need configuration.

    '''

    return np.random.randn(my_array_size, my_array_size)


def my_other_annotated_function(
        my_input_parameter: Annotated[ArrayLike, 'name:foo']
    ) -> Annotated[ArrayLike, 'name:bar']:
    '''this is another annotated function

    Same as `my_other_plain_function()` but knows that its argument has
    default name "foo" and that its return value has default name "bar".

    '''

    return -my_input_parameter


# instead of always writing out these annotations, you can define aliases
Foo = Annotated[ArrayLike, 'name:foo']
Bar = Annotated[ArrayLike, 'name:bar']
Baz = Annotated[None, 'name:baz']


def my_annotated_print_function(
        my_array: Foo,
        my_other_array: Bar,
        my_name: str = 'an array',
        my_other_name: str = 'its negative'
    ) -> Baz:
    '''this function prints its arguments

    This function knows all of its arguments and return name by its
    annotations (using aliases) and default values.  No configuration
    necessary!

    '''

    print(my_name)
    print(my_array)
    print(my_other_name)
    print(my_other_array)
