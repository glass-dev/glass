
How GLASS works
===============

The core idea of GLASS is that large simulations are best performed
*iteratively*.  Instead of simulating e.g. all matter or all galaxies in the
universe up to some maximum redshift, GLASS generates the simulation in thin,
nested shells, one after another.

Generators
----------

Within the code, the idea of an iterative simulation is encapsulated in the
*generator* concept.  The simulation itself is essentially one big loop, which
at every iteration calls a list of *generators* in turn.  Every generator
receives a set of inputs and yields a set outputs.  All work is hence done by
the generators; the simulation itself merely keeps track of their inputs and
outputs.

.. mermaid::

    flowchart LR
        A(simulation) --> B{iterate}
        B --> C{generators}
        C --> I[send inputs]
        I --> G[run generator]
        G --> O[store outputs]
        O --> C
        C --> B

Generators are thus conceptually very simple routines that only need to know how
to receive a set of inputs and produce a set of outputs.  However, they usually
require, or at least benefit from, having memory of their runs (i.e. *state*).
GLASS deals with this using coroutines.


Coroutines
----------

Coroutines are fundamentally functions that can be suspended and resumed.  In
essence, this means that a coroutine can run for a while, e.g. to produce an
output, and hand control back to its caller.  When called again, the coroutine
will resume, and remember its local context (i.e. *state*).

The GLASS generators are coroutines that also contain essentially just one big
loop.  Inside, they *i) yield* an output (``None`` at first, since nothing has
been computed yet), *ii) receive* new inputs for the next iteration, and *iii)
compute* the next result.

.. mermaid::

    flowchart LR
        A(generator) --> B{loop}
        B --> C[yield]
        C -.->|suspend| D[receive]
        D --> E[compute]
        E --> B

Besides the loop, coroutines can perform some initial computation, as well as a
finalisation step after for the loop when the simulation terminates.


Implementation
--------------

If this all sounds a little abstract and mysterious: Coroutines in Python are in
fact a well-known concept under the guise of *Python generators*::

    def song():
        warm_up()
        yield 'hickory dickory dock'
        # ---
        breathe()
        yield 'the mouse ran up the clock'
        # ---
        cool_down()

    for verse in song():
        sing(verse)

When called, the function ``song()`` executes until the first ``yield``
statement, and is suspended.  This allows the ``for`` loop to run and process
the resulting value.  The function then resumes in the next iteration, performs
some computation, and yields the next result.  Python generators are thus
coroutines, and GLASS generators are essentially just fancy Python generators --
hence the name!

The only difference between the trivial example above and a real GLASS generator
is that the letter also receives inputs.  Python generators receive input simply
by assigning the yield statement::

    def echo(num=2):
        print('entering echo chamber')
        # initial yield is always None
        echoed = None
        # loop until the generator is told to stop
        # at every loop, yield the echoed words, and receive a new word
        while True:
            try:
                word = yield echoed
            except GeneratorExit:
                break

            echoed = ' '.join([word]*num)
        print('leaving echo chamber')

This is essentially a complete GLASS generator.  To send inputs, the standard
``.send()`` mechanism of the Python generator is used::

    # this instantiates the generator, but no code is executed yet
    g = echo(3)
    # this primes the generator and runs until the first yield
    g.send(None)  # output: entering echo chamber
    # now we send some words to the generator and print the results:
    for word in ['hello', 'world']:
        echoed = g.send(word)
        print(echoed)  # output: hello hello hello // world world world
    # tell the generator we are done
    g.close()  # output: leaving echo chamber
