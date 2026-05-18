# GLASS benchmarks on Archer2

For a more consistent benchmarking and regression testing environment we have
been trialing using the UCL machine [Archer2](https://www.archer2.ac.uk/).

## Setting up

1. **Install uv:** Firstly, install uv via curl

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. **Clone GLASS:** Clone the glass repo into the work section of Archer2 -
   `/work/<budget-code>/<budget-code>/<user-id>`

   ```sh
   cd /work/<budget-code>/<budget-code>/<user-id>
   git clone https://github.com/glass-dev/glass.git
   export GLASS_DIR="/work/<budget-code>/<budget-code>/<user-id>/glass"
   ```

3. **Run the benchmarks script:** Now we have cloned glass we can run the script
   `run_benchmarks.sh` which will setup the required environments and submit
   regression test job to slurm.
