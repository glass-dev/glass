#!/bin/bash --login
# shellcheck disable=SC1091

#SBATCH --job-name=glass_reg_test_cpu
#SBATCH --output=%x-%j.out
#SBATCH --nodes=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=1
#SBATCH --time=0:30:0
#SBATCH --partition=standard
#SBATCH --qos=standard

# Ensure uv is available
source "${HOME/home/work}/.profile" # HOME starts as /home/... but uv needs to be on /work/...

# Recommended environment settings
# Stop unintentional multi-threading within software libraries
export OMP_NUM_THREADS=1
# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Set path to class and select base or head
export GLASS_DIR="$(pwd)"
export REGRESSION_DIR="$GLASS_DIR/tests/regression"
export REGRESSION_OUTPUT_PATH="$REGRESSION_DIR/archer2/.benchmarks"
export REGRESSION_SHARED_FLAGS=(
  "--benchmark-storage=file://$REGRESSION_OUTPUT_PATH"
  "--benchmark-calibration-precision=1000"
  "--benchmark-columns=mean,stddev,rounds"
  "--benchmark-max-time=5.0"
  "--benchmark-sort=name"
  "--benchmark-timer=time.process_time"
)
START_VENV_BIN="$GLASS_DIR/.venv-start/bin"
END_VENV_BIN="$GLASS_DIR/.venv-end/bin"

# Remove old regression results
rm -rf "$REGRESSION_OUTPUT_PATH"

# Change into archer2 dir to ensure we don't pickup the glass directory as an import
cd "$REGRESSION_DIR/archer2" || exit

# Generate the base report for comparison later
source "$START_VENV_BIN/activate"
srun "$START_VENV_BIN/python" -m pytest "$REGRESSION_DIR" \
    --benchmark-autosave "${REGRESSION_SHARED_FLAGS[@]}"
deactivate

# Run the stable and unstable regression tests and compare to the base ref
source "$END_VENV_BIN/activate"
srun "$END_VENV_BIN/python" -m pytest "$REGRESSION_DIR" -m stable \
    --benchmark-compare=0001 \
    --benchmark-compare-fail=mean:5% "${REGRESSION_SHARED_FLAGS[@]}"
srun "$END_VENV_BIN/python" -m pytest "$REGRESSION_DIR" -m unstable \
    --benchmark-compare=0001 \
    --benchmark-compare-fail=mean:0.0005 "${REGRESSION_SHARED_FLAGS[@]}"
