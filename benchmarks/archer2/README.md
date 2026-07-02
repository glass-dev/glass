# Archer2

This document contains instructions specific to running the glass benchmarks on
Archer2

## Prerequisites

Archer2 only support rocm up to v0.6.x. Therefore, we are restricted to using
`jax==0.4.35`. This in turn restricts us to using python3.12. The relevant
modules can be loaded via the provided script [setup-env.sh](./setup-env.sh)

> Note that setup-env.sh is called automatically by the submission script
> [submit-gpu.sh](./submit-gpu.sh)

## Running the benchmarks

Benchmarks should be submitted as a batch job to slurm via the provided script.

For example to benchmark using jax with amd/rocm, run the following from the
root of the glass repo on Archer2:

```sh
sbatch benchmarks/archer2/submit-gpu.sh -d "$(pwd)" -x jax --healpy-datapath "$HEALPY_DATAPATH"
```

> To understand what HEALPY_DATAPATH is, read an explanation in
> [benchmarks/README.md#healpy-data](../README.md#healpy-data)

## Setting up UV on Archer2

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
