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
   install the required python dependencies which have been extracted from the
   [pyproject.toml](../../../pyproject.toml) into
   [test-requirements.txt](./test-requirements.txt). Therefore, we must create
   our virtual environment...

   ```sh
   python3 -m venv $GLASS_DIR/.venv # This only needs to be done once
   source $GLASS_DIR/.venv/bin/activate
   python -m pip install -r $GLASS_DIR/tests/benchmarks/myriad/test-requirements.txt
   ```

3. Now that our environment is setup, we can submit our regression test script
   ([run_regression_test.sh](./run_regression_test.sh)) to the scheduler...

   ```sh
   qsub run_regresion_test.sh
   ```
