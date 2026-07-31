==============
Running on GPU
==============

GLASS is fully `Python Array API <https://data-apis.org/array-api/latest/>`_
compatible as of version
`v2026.2 <https://github.com/glass-dev/glass/releases/tag/v2026.2>`_. Thus, it is
possible to pass arrays from any Array API compatible backends. Therefore,
running GLASS on GPU is as simple as selecting an array backend which is both
Array API compatible and GPU enabled. For example, if JAX is installed with
the relevant optional dependencies it will by default utilise any GPU devices
it has available.

Unfortunately, we have found that this does not necessarily mean running GLASS
with GPU enabled JAX will give an immediate performance boost. The performance
gains, or as is often the case, degradations of running GLASS on GPU depends on
many factors including the size of the problem and the accelerator architecture
available.

For examples of running GLASS on GPU, check out our
`benchmarks <https://github.com/glass-dev/glass/tree/main/benchmarks>`_.
