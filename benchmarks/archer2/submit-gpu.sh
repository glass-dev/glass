#!/bin/bash --login
# shellcheck disable=SC1091

#SBATCH --job-name=glass_benchmark_gpu
#SBATCH --output=%x-%j.out
#SBATCH --cpus-per-task=1
#SBATCH --nodes=1
#SBATCH --gpus=1
#SBATCH --time=0:30:0
#SBATCH --partition=gpu
#SBATCH --qos=gpu-shd

# Setup environment
./setup-env.sh

# Recommended environment settings
# Stop unintentional multi-threading within software libraries
export OMP_NUM_THREADS=1
# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

GLASS_DIR=""
ARRAY_BACKEND="numpy"
HEALPY_DATAPATH=""

help() {
  echo "Usage:"
  echo "    $0 -d <glass/dir> [-x <array_backend>] [-h|--help]"
  echo ""
  echo "ARGS:"
  echo "    -h | --help                          Display this help message."
  echo "    -d | --glass-dir <glass/dir>         Path to the cloned glass directory."
  echo "    -x | --array-backend <array_backend> The array backend to use for the benchmarks."
  echo "                                         Defaults to NumPy."
  echo "    --healpy-datapath <healpy-datapath>  The path to the healpy-data repo to allow"
  echo "                                         running offline."
}

# Ensure uv is available
source "${HOME/home/work}/.profile" # HOME starts as /home/... but uv needs to be on /work/...

# Stop unintentional multi-threading within software libraries
export OMP_NUM_THREADS=1
# Ensure the cpus-per-task option is propagated to srun commands
export SRUN_CPUS_PER_TASK=$SLURM_CPUS_PER_TASK

# check for no input arguments and show help
if [ $# -eq 0 ];
then
    help
    exit 1
fi

while [ $# -gt 0 ] ; do
    case $1 in
        -h | --help)
            help
            exit 0
            ;;
        -d | --glass-dir)
            GLASS_DIR="$2"
            shift 2
            continue
            ;;
        -x | --array-backend)
            ARRAY_BACKEND="$2"
            shift 2
            continue
            ;;
        --healpy-datapath)
            HEALPY_DATAPATH="$2"
            shift 2
            continue
            ;;
        *)
            echo "Invalid option: $1" >&2;
            help
            exit 1
            ;;
    esac
    shift 1
done

if [[ "$GLASS_DIR" == "" ]]
then
  echo "GLASS_DIR must be provided"
  help
  exit 1
fi

# Get execution method
PYTHON_COMMAND="$GLASS_DIR/.venv/bin/python"
if [[ -n "$SLURM_JOB_ID" || -n "$SLURM_BATCH_SCRIPT" ]]; then
  echo "Running under SLURM (batch). SLURM_JOB_ID=${SLURM_JOB_ID:-unknown}"
  PYTHON_COMMAND="srun $PYTHON_COMMAND"
else
  echo "Running directly from CLI"
fi

HEALPY_DATAPATH="$HEALPY_DATAPATH" ARRAY_BACKEND="$ARRAY_BACKEND" $PYTHON_COMMAND benchmarks/lensing.py
