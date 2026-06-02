# GLASS benchmarks on Archer2

For a more consistent benchmarking and regression testing environment we have
been trialing using the UCL machine [Archer2](https://www.archer2.ac.uk/).

## Setting up

1. **Install uv:** Firstly, install uv via curl onto the `/work` partition

   ```sh
   cd "${HOME/home/work}"
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then we must make sure uv is available on the login node and the worker node.
   To do this we can update our start up scripts (`.profile`) on both
   partitions. Therefore, save the following into `$HOME/.profile`

   ```sh
   WORK_DIR="${HOME/home/work}"
   cd "$WORK_DIR"
   source "$WORK_DIR/.profile"
   ```

   and then save the following into `${HOME/home/work}/.profile`.

   ```sh
   export HOME="${HOME/home/work}"
   source "$HOME/.local/bin/env"
   ```

   Now when you next login to archer2, uv will be in your path and you will be
   on the `/work` partition as your `HOME` dir.

2. **Clone GLASS:** Clone the glass repo into the `/work` partition of Archer2 -
   `/work/<budget-code>/<budget-code>/<user-id>`

   ```sh
   cd "${HOME/home/work}"
   git clone https://github.com/glass-dev/glass.git
   ```

## Run the regression tests

Now we have cloned glass, we can run the script
[run_regression_test.sh](./run_regression_test.sh) which will setup the required
environments and submit regression test job to slurm. A help message is
provided. Just run `run_regression_test.sh -h` from the root of the GLASS repo.
