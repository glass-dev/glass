======================================
Core functionality (:mod:`glass.core`)
======================================

.. currentmodule:: glass.core

This module contains the core functions of GLASS.

Generation
==========

.. autofunction:: generate
.. autoexception:: GeneratorError
   :members: generator, state


Generator groups
================

Generators can be combined into a logical unit using the :func:`group`
generator.  The outputs of generators in a group are stored inside the group.
Generators in a group will receive input variables from within their group
first, before looking at the parent of the group.  This can be used to obtain
the same variables from multiple generators.  Groups can be nested.

.. seealso::

    :doc:`examples:basic/groups` example
        For an example of how groups can be used.

.. autofunction:: group
