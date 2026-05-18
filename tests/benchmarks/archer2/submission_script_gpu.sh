#!/bin/bash --login

#SBATCH --job-name=glass_gpu_benchmarks
#SBATCH --cpus-per-task=1
#SBATCH --gpus=1
#SBATCH --time=0:30:0

# Replace [budget code] below with your project code (e.g. t01)
#SBATCH --account=[budget code]
#SBATCH --partition=standard
#SBATCH --qos=standard

# Load GPU modules
module load PrgEnv-amd/8.4.0
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan

# Recommended environment settings
# Stop unintentional multi-threading within software libraries
export OMP_NUM_THREADS=1
# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# Set path to class and select base or head
GLASS_DIR="$1"
BENCHMARKS_DIR="$GLASS_DIR/tests/benchmarks"
BENCHMARKS_SHARED_FLAGS=(
  "--benchmark-storage=file://$BENCHMARKS_DIR/outputs"
  "--benchmark-calibration-precision=1000"
  "--benchmark-columns=mean,stddev,rounds"
  "--benchmark-max-time=5.0"
  "--benchmark-sort=name"
  "--benchmark-timer=time.process_time"
)

# Generate the base report for comparison later
source "$GLASS_DIR/.venv-base/bin/activate"
srun uv run pytest "$BENCHMARKS_DIR" --benchmark-autosave "${BENCHMARKS_SHARED_FLAGS[@]}"
deactivate

# Run the stable and unstable benchmarks and compare to the base ref
source "$GLASS_DIR/.venv-head/bin/activate"
srun uv run pytest "$BENCHMARKS_DIR" -m stable --benchmark-compare=0001 --benchmark-compare-fail=mean:5% "${BENCHMARKS_SHARED_FLAGS[@]}"
srun uv run pytest "$BENCHMARKS_DIR" -m unstable --benchmark-compare=0001 --benchmark-compare-fail=mean:0.0005 "${BENCHMARKS_SHARED_FLAGS[@]}"
