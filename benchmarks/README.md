# GLASS Benchmarks

These benchmarks are intended to allow benchmarking GLASS on various machines,
architectures and node configurations.

Before running any benchmarks, you must first setup your venv with the required
dependencies. For all backends, this will require installing the dependency
group `benchmarks`, like so,

```sh
uv sync --group benchmarks
```

However, it may also be necessary to install other dependencies / dependency
groups. For example, read the
[prerequisites for Archer2](./archer2/README.md#prerequisites)

## Adding a new benchmark

Any new benchmarks should be added as new `.py` file in the `benchmarks`
directory.

To create a benchmark, add a new `,py` file in the `benchmarks` directory and
write a function you wish to benchmark. Then pass this function, along with any
arg/kwargs, into `benchmark_utils.run_benchmark`.

The example below shows how to write a benchmark for the function
`function_to_benchmark`, which runs for all supported array backends:

```py
from typing import TYPE_CHECKING
from benchmark_utils import run_benchmark, xp_available_backends

if TYPE_CHECKING:
    from types import ModuleType
    from glass._types import FloatArray

for xp in xp_available_backends.values():
    # Do some setup which I do not want to time
    x = xp.asarray([1, 2, 3, 4, 5, 6, 7, 8, 9], dtype=xp.float64)
    y = xp.asarray([11, 12, 13, 14, 15, 16, 17, 18, 19], dtype=xp.float64)

    def function_to_benchmark(
        *,
        x: FloatArray,
        y: FloatArray,
        xp: ModuleType,
    ) -> FloatArray:
        """An example function to be benchmarked."""
        return x / xp.vecdot(x, y)

    run_benchmark(
        function_to_benchmark,
        x=x,
        y=y,
        xp=xp,
    )
```

##  healpy-data

glass depends on data from the healpy-data repo. If not found locally, glass
downloads this from the internet. Since most clusters will not allow internet
access from a worker node. We must provide a local copy of this data before
submitting a job. To download this data we can use git:

```sh
git clone --depth 1 https://github.com/healpy/healpy-data
export HEALPY_DATAPATH="$(pwd)/healpy-data"
```
