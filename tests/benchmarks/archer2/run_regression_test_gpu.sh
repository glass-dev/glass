#!/bin/bash --login

#SBATCH --job-name=test_job
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0:30:0

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]
#SBATCH --partition=standard
#SBATCH --qos=standard

GLASS_DIR=""
GLASS_REPO_URL="https://github.com/glass-dev/glass"
BENCHMARKS_DIR="$GLASS_DIR/tests/benchmarks"
BENCHMARKS_SHARED_FLAGS=(
  "--benchmark-storage=file://$BENCHMARKS_DIR/outputs"
  "--benchmark-calibration-precision=1000"
  "--benchmark-columns=mean,stddev,rounds"
  "--benchmark-max-time=5.0"
  "--benchmark-sort=name"
  "--benchmark-timer=time.process_time"
)

# Load GPU modules
module load PrgEnv-amd/8.4.0
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan

# Activate the correct environment
source "$GLASS_DIR/.venv/bin/activate"

# Recommended environment settings
# Stop unintentional multi-threading within software libraries
export OMP_NUM_THREADS=1
# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Set the base and head refs to be compared
BASE_REF="<some-git-commit-hash>"
HEAD_REF="<some-git-commit-hash>"

# Remove old benchmark results
rm -rf "$BENCHMARKS_DIR/outputs"

# Make sure to installation of glass already exists
python -m pip uninstall glass -y

# Install the base ref of glass
python -m pip install "git+$GLASS_REPO_URL@$BASE_REF"

# Generate the base report for comparison later
ARRAY_BACKEND=jax srun --ntasks=1 --cpus-per-task=1 python -m pytest "$BENCHMARKS_DIR" --benchmark-autosave "${BENCHMARKS_SHARED_FLAGS[@]}"

# Uninstall the base ref of glass
python -m pip uninstall glass -y

# Install the head ref of glass
python -m pip install "git+$GLASS_REPO_URL@$HEAD_REF"

# Run the stable and unstable benchmarks and compare to the base ref
srun python -m pytest "$BENCHMARKS_DIR" -m stable --benchmark-compare=0001 --benchmark-compare-fail=mean:5% "${BENCHMARKS_SHARED_FLAGS[@]}"
srun python -m pytest "$BENCHMARKS_DIR" -m unstable --benchmark-compare=0001 --benchmark-compare-fail=mean:0.0005 "${BENCHMARKS_SHARED_FLAGS[@]}"
