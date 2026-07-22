#!/bin/bash -l
# shellcheck disable=SC1091

#SBATCH -J glass_benchmark_gpu
#SBATCH -o %x-%j.out
#SBATCH -e %x-%j.err
#SBATCH -p SH5_MI300A_SPX
#SBATCH --exclusive
#SBATCH -t 3:0:0
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gpus=1

module load rocm

# Recommended environment settings
# Stop unintentional multi-threading within software libraries
export OMP_NUM_THREADS=1

GLASS_DIR=""
ARRAY_BACKEND="numpy"
HEALPY_DATAPATH=""

help() {
  echo "Usage:"
  echo "    $0 -d <glass/dir> [-x <array_backend>] [--healpy-datapath <healpy-datapath>] [-h|--help]"
  echo ""
  echo "ARGS:"
  echo "    -h | --help                          Display this help message."
  echo "    -d | --glass-dir <glass/dir>         Path to the cloned glass directory."
  echo "    -x | --array-backend <array_backend> The array backend to use for the benchmarks."
  echo "                                         Defaults to NumPy."
  echo "    --healpy-datapath <healpy-datapath>  The path to the healpy-data repo to allow"
  echo "                                         running offline. Defaults to <glass/dir>/healpy-data"
}

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

# Ensure GLASS_DIR is provided
if [[ "$GLASS_DIR" == "" ]]
then
  echo "GLASS_DIR must be provided"
  help
  exit 1
fi

# Set HEALPY_DATAPATH default
if [[ "$HEALPY_DATAPATH" == "" ]]; then
    HEALPY_DATAPATH="$GLASS_DIR/healpy-data"
fi

# Flags to maximise jax gpu performance
#export JAX_ENABLE_PGLE=true
#export XLA_FLAGS="--xla_gpu_enable_latency_hiding_scheduler=true"

export ARRAY_BACKEND="$ARRAY_BACKEND"
export HEALPY_DATAPATH="$HEALPY_DATAPATH"

# Run benchmarks with shared memory on and off
for i in {1,0}
do
    echo "Running benchmarks with HSA_XNACK=$i"
    export HSA_XNACK=$i

    for n in {128,256,512,1024}
    do
        echo "Running benchmark with nside/lmax = $n"
        sed -i -E "s/nside = lmax = [0-9]+/nside = lmax = $n/g" "$GLASS_DIR/benchmarks/lensing.py"

        # Run benchmark via slurm
        "$GLASS_DIR/.venv/bin/python" benchmarks/lensing.py
    done
done
