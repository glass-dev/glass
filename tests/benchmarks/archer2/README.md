# GLASS benchmarks on Archer2

For a more consistent benchmarking and regression testing environment we have
been trialing using the UCL machine [Archer2](https://www.archer2.ac.uk/).

## Setting up

1. Firstly, install uv via curl

   ```sh
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

2. Clone the glass repo into the work section of Archer2 -
   `/work/<budget-code>/<budget-code>/<user-id>`

   ```sh
    cd /work/<budget-code>/<budget-code>/<user-id>
   git clone https://github.com/glass-dev/glass.git
   export GLASS_DIR="/work/<budget-code>/<budget-code>/<user-id>/glass"
   ```

3. Then, to reduce the required run time of the benchmark jobs on Archer2, we
   install the required dependencies from the groups `test` and `archer2`:

   ```sh
   cd $GLASS_DIR
   uv venv
   uv sync --group test --group archer2
   ```

4. Now that our environment is setup, before we can submit our regression test
   script to the scheduler, we must make some changes to the submission script -
   ([run_regression_test.sh](./run_regression_test.sh)). These changes will
   include setting the path to the root of the glass repo on archer2 and
   choosing the BASE and the HEAD refs to be compared in the regression test.
   Once we have done this, we can submit the job...

   ```sh
   sbatch $GLASS_DIR/tests/benchmarks/archer2/run_regresion_test.sh
   ```
