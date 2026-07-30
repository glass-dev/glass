# AAC6

This document contains instructions specific to running the glass benchmarks on
[AAC6](https://aac.amd.com/help/).

## Prerequisites

As AAC6 is a managed cluster, there are several steps to setting up your
environment. First you should
[setup uv](https://docs.astral.sh/uv/getting-started/installation/). Then you
will need to install you python virtual environment. Finally, you will need to
load specific modules for the GPU benchmark. Your virtual environment can be
created via:

```sh
uv venv
```

### CPU prerequisites

To setup your python environment for running the cpu benchmark on AAC6, run the
following commands.

```sh
uv sync --group benchmarks
```

### GPU prerequisites

For the gpu benchmark, there is an additional dependency group `aac6-gpu` which
includes on `benchmarks`:

```sh
uv sync --group aac6-gpu
```

## Running the benchmarks

Benchmarks can be submitted as a batch job to slurm via the provided script.

For example to benchmark using jax with amd/rocm, run the following from the
root of the glass repo on AAC6. You will need to make some changes to the
submissions script (updating your email, budget code, etc):

```sh
sbatch benchmarks/aac6/submit-gpu.sh -d "$(pwd)" -x jax --healpy-datapath "$HEALPY_DATAPATH"
```

> To understand what HEALPY_DATAPATH is, read an explanation in
> [benchmarks/README.md#healpy-data](../README.md#healpy-data)

This script will attempt to run a range of benchmarks with different nside
values for the [lensing benchmark](../lensing.py). It is intended as more of an
example of what can be done and how to submit such a script rather than defining
the exact "ideal" benchmark.

[benchmarks/aac6/submit-gpu.sh](./benchmarks/aac6/submit-gpu.sh) specifically
will submit a job to the SH5_MI300A_SPX queue to utilise shared CPU/GPU memory.
