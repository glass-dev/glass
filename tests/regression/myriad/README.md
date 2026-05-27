# GLASS benchmarks on Myriad

For a more consistent benchmarking and regression testing environment we have
been trialing using the UCL machine
[Myriad](https://www.rc.ucl.ac.uk/docs/Clusters/Myriad/).

To simplify running benchmarks on myriad there are several scripts provided.

## Setting up

1. Firstly, we must load the necessary modules, this can be done using the file
   [load_modules.sh](./load_modules.sh)...

   ```sh
   export GLASS_DIR=/path/to/glass
   source $GLASS_DIR/tests/benchmarks/myriad/load_modules.sh
   ```

2. Then, to reduce the required run time of the benchmark jobs on myriad, we
   install the required python dependencies from the group `test`. Therefore, we
   must create our virtual environment, ensuring we upgrade pip to a version
   which supports dependency groups...

   ```sh
   python3 -m venv $GLASS_DIR/.venv
   source $GLASS_DIR/.venv/bin/activate
   python -m pip install --upgrade pip
   ```

   Now we can install our dependencies...

   ```sh
   python -m pip install --group test
   ```

3. Now that our environment is setup, before we can submit our regression test
   script to the scheduler, we must make some changes to the submission script -
   ([run_regression_test.sh](./run_regression_test.sh)). These changes will
   include setting the path to the root of the glass repo in myriad and choosing
   the BASE and the HEAD refs to be compared in the regression test. Once we
   have done this, we can submit the job...

   ```sh
   qsub $GLASS_DIR/tests/benchmarks/myriad/run_regresion_test.sh
   ```
