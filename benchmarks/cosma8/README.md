# Cosma8

This document contains instructions specific to running the glass benchmarks on
[cosma8](https://cosma.readthedocs.io/en/latest/cosma8.html)

## Prerequisites

As cosma8 is a managed cluster, there are several steps to setting up your
environment. First you should
[setup uv](https://docs.astral.sh/uv/getting-started/installation/). Then you
will need to install you python virtual environment:

```sh
uv venv
```

### CPU prerequisites

To setup your python environment for running the cpu benchmark on cosma8, run
the following commands.

```sh
uv sync --group benchmarks
```

### GPU prerequisites

For the GPU benchmark, there is an additional dependency group `jax-rocm7` which
includes `benchmarks`:

```sh
uv sync --group jax-rocm7
```

## Running the benchmarks

### ROCm

Benchmarks can be submitted as a batch job to slurm via the provided script. For
example to benchmark using JAX with AMD/ROCm, run the following from the root of
the glass repo on cosma8. You will need to make some changes to the submissions
script (updating your email, budget code, etc):

```sh
sbatch benchmarks/cosma8/submit-gpu.sh -d "$(pwd)" -x jax --healpy-datapath "$HEALPY_DATAPATH"
```

> To understand what HEALPY_DATAPATH is, read an explanation in
> [benchmarks/README.md#healpy-data](../README.md#healpy-data)

This script will attempt to run a range of benchmarks with different nside
values for the [lensing benchmark](../lensing.py). It is intended as more of an
example of what can be done and how to submit such a script rather than defining
the exact "ideal" benchmark.

[benchmarks/cosma8/submit-gpu.sh](./submit-gpu.sh) specifically will submit a
job to the mi300x queue. However, if you wish to run on a GPU with shared memory
architecture, you will need to ssh onto the ga008 partition:

```sh
ssh ga008
```

> Note that `ga008` is a shared resource so benchmarking there is unlikely to
> give reliable results:

Access to these different partitions/queues must be requested via the DiRAC
SAFE. For further information
[read the docs](https://cosma.readthedocs.io/en/latest/account.html).

### CUDA

Cosma8 provides some direct ssh access nodes for testing CUDA applications - see
the [docs](https://cosma.readthedocs.io/en/latest/gpu.html).

Fist install your dependencies

```sh
uv venv
uv sync --group benchmarks
uv pip install jax[cuda13]
```

Then setup your env on the GPU node

```sh
ssh gn005
cd glass
export XLA_PYTHON_CLIENT_PREALLOCATE=false
export ARRAY_BACKEND=jax
export HEALPY_DATAPATH="$(pwd)/healpy-data"
```

Finally, run the profile:

```sh
RUN_PROFILE=True uv run python benchmarks/lensing.py
```
