# GLASS regression testing on Archer2

For a more consistent regression testing environment we have been trialling using
[Archer2](https://www.archer2.ac.uk/).

## Setting up

1. **Install uv:** Firstly, install uv via curl onto the `/work` partition

   ```sh
   cd "${HOME/home/work}"
   curl -LsSf https://astral.sh/uv/install.sh | sh
   ```

   Then we must make sure uv is available on the login node and the worker node.
   To do this we can update our startup scripts (`.profile`) on both partitions.
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

   Now when you next log in to archer2, uv will be in your path, and you will be
   on the `/work` partition as your `HOME` dir.

2. **Clone GLASS:** Clone the GLASS repo into the `/work` partition of Archer2:

   ```sh
   cd "${HOME/home/work}"
   git clone https://github.com/glass-dev/glass.git
   ```

## Run the regression tests

Now we have cloned glass, we can run the script
[run_regression_test.sh](./run_regression_test.sh) which will set up the required
environments and submit regression test job to slurm. A help message is
provided. Just run `./tests/regression/archer2/run_regression_test.sh -h` from
the root of the GLASS repo.

### Example execution

If you wished to run a test to check for regressions from `main` to a feature
branch called `feature` using the budget from account code `ecsega23`, you could
run the following command from the root of the GLASS repo.

```sh
./tests/regression/archer2/run_regression_test.sh \
   -d "$(pwd)"                                    \
   -s main                                        \
   -e feature                                     \
   -a ecsega23
```

> Note that this script does not have to be run from the root of the GLASS repo.

#### The results

Once you have run the script `run_regression_test.sh`, you will be given a job
ID such as `13925816`, which you can find in the terminal output within a line
which looks like the following output.

```txt
Submitted batch job 13925816
```

You can monitor your submitted jobs and their status in the slurm queue via:

```sh
squeue -u $USER
```

Once your job has finished, you can inspect the results by reading the outputted
file which, in the above case, would be called `glass_reg_test_cpu-13925816.out`

#### Troubleshooting

- If your job does not finish in time your output will contain lines such as,

  <!-- markdownlint-disable MD013 -->

  ```txt
  slurmstepd: error: *** STEP 13925937.0 ON nid001228 CANCELLED AT 2026-06-03T11:12:41 DUE TO TIME LIMIT ***
  slurmstepd: error: *** JOB 13925937 ON nid001228 CANCELLED AT 2026-06-03T11:12:41 DUE TO TIME LIMIT ***
  ```

  <!-- markdownlint-enable MD013 -->

  To rectify this issue, you can increase the time limit of your job by altering
  the following line in [submission_script_cpu.sh](./submission_script_cpu.sh)

  ```sh
  #SBATCH --time=0:30:0
  ```
