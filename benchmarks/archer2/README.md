# ARCHER2

This document contains instructions specific to running the glass benchmarks on
[ARCHER2](https://www.archer2.ac.uk/)

## Prerequisites

As ARCHER2 is a managed cluster, there are several steps to setting up your
environment. First you should [setup uv](#setting-up-uv-on-archer2). Then you
will need to install you python virtual environment. Finally, you will need to
load specific modules for the GPU benchmark.

Note that for the GPU benchmarks ARCHER2 only support rocm up to v0.6.x.
Therefore, we are restricted to using `jax==0.4.35`. This in turn restricts us
to using python 3.12. Thus, to produce a useful CPU vs GPU comparison create
your venv using the following command

```sh
uv venv --python 3.12
```

### CPU prerequisites

To setup your python environment for running the cpu benchmark on ARCHER2, run
the following commands.

```sh
uv sync --group benchmarks
uv pip install --no-deps glass-ext-camb
```

### GPU prerequisites

For the gpu benchmark, there is an additional dependency group `archer2-gpu`
which includes on `benchmarks`:

```sh
uv sync --group archer2-gpu
uv pip install --no-deps glass-ext-camb
```

Once your python environment is setup you must load the relevant modules via the
provided script [setup-gpu-env.sh](./setup-gpu-env.sh).

> Note that many of the modules loaded are only available once you are running
> on a GPU worker node. However, `setup-gpu-env.sh` is called automatically by the
> submission script [submit-gpu.sh](./submit-gpu.sh).

## Running the benchmarks

Benchmarks should be submitted as a batch job to slurm via the provided script.
For example to benchmark using JAX with AMD/ROCM, run the following from the
root of the glass repo on ARCHER2. You will need to make some changes to the
submissions script (updating your email, budget code, etc.):

```sh
sbatch benchmarks/archer2/submit-gpu.sh -d "$(pwd)" -x jax --healpy-datapath "$HEALPY_DATAPATH"
```

> To understand what `HEALPY_DATAPATH` is, read an explanation in
> [benchmarks/README.md#healpy-data](../README.md#healpy-data)

This script will attempt to run a range of benchmarks with different nside
values for the [lensing benchmark](../lensing.py). It is intended as more of an
example of what can be done and how to submit such a script rather than defining
the exact "ideal" benchmark.

[benchmarks/archer2/submit-gpu.sh](./benchmarks/archer2/submit-gpu.sh)
specifically will submit a job to the archer2 amd gpu testbed queue.

## Setting up uv on ARCHER2

Firstly, install uv via curl onto the `/work` partition

```sh
cd "${HOME/home/work}"
curl -LsSf https://astral.sh/uv/install.sh | sh
```

Then we must make sure uv is available on the login node and the worker node. To
do this we can update our start up scripts (`.profile`) on both partitions.
Therefore, execute the following

```sh
cat <<'EOF' >> "$HOME/.profile"
WORK_DIR="${HOME/home/work}"
cd "$WORK_DIR"
source "$WORK_DIR/.profile"
EOF
```

and similarly

```sh
cat <<'EOF' >> "${HOME/home/work}/.profile"
export HOME="${HOME/home/work}"
source "$HOME/.local/bin/env"
EOF
```

Now when you next login to archer2, uv will be in your path and you will be on
the `/work` partition as your `HOME` dir.
