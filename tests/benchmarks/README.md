# GLASS Benchmarks

These benchmarks are intended to allow benchmarking GLASS on various machines, architectures and node configurations.

## Running the benchmarks

To run the benchmarks we must setup our machine specific environment.
Therefore, run the following from the root of the glass repo:

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

Now you should be able to run the benchmarks

```sh
.venv-benchmark/bin/python -m pytest tests/benchmarks \
    --benchmark-autosave                              \
    --benchmark-columns=mean,stddev,rounds,iterations \
    --benchmark-max-time=5.0                          \
    --benchmark-sort=name                             \
    --benchmark-timer=time.process_time 
```