# GLASS Benchmarks

These benchmarks are intended to allow benchmarking GLASS on various machines,
architectures and node configurations.

Before running any benchmarks, you must first setup your venv with the required
dependencies.

##  healpy-data

glass depends on data from the healpy-data repo. If not found locally, glass
downloads this from the internet. Since most clusters will not allow internet
access from a worker node. We must provide a local copy of this data before
submitting a job. To download this data we can use git:

```sh
git clone --depth 1 https://github.com/healpy/healpy-data
export HEALPY_DATAPATH="$(pwd)/healpy-data"
```
