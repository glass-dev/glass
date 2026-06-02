# GLASS Benchmarks

These benchmarks are intended to allow benchmarking GLASS on various machines,
architectures and node configurations.

## Running the benchmarks

To run the benchmarks we must setup our machine specific environment. Therefore,
run the following from the root of the glass repo:

```sh
source benchmarks/environments/<machine-name>.sh
```

Then, install the neccessary python dependencies:

> Note: You will need to setup uv on your machine.

```sh
uv venv .venv-benchmark
source .venv-benchmark/bin/activate
uv sync --active --group benchmarks
```

Now you should be able to run the benchmarks with the following command.

```sh
.venv-benchmark/bin/python -m pytest tests/benchmarks \
    --benchmark-autosave                              \
    --benchmark-columns=mean,stddev,rounds,iterations \
    --benchmark-max-time=5.0                          \
    --benchmark-sort=name                             \
    --benchmark-timer=time.process_time
```

### Benchmarking a cluster

To get a reliable benchmark, you should run your job on an exlusive node of the
machine you are benchmarking. You can do this either via an interactive session
or through submitting a job to the queue system.

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
