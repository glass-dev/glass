#!/bin/bash -l

GLASS_DIR=""
GLASS_REPO_URL="https://github.com/glass-dev/glass"
CPU_OR_GPU="cpu"
START_REF=""
END_REF=""
START_VENV=".venv-start"
END_VENV=".venv-end"
ACCOUNT=""

help() {
  echo "Usage:"
  echo "    $0 "
  echo ""
  echo "ARGS:"
  echo "    -h | --help                      Display this help message."
  echo "    -d | --glass-dir <glass/dir>     Path to the cloned glass directory."
  echo "    -s | --start-ref <start_ref>     The git ref to be used as the initial state."
  echo "    -e | --end-ref <end_ref>         The git ref to be used as the final state."
  echo "    -a | --account <archer2_account> The archer2 account code to run jobs against."
  echo "    -g | --gpu                       Flag to state the gpu benchmark should be ran."
}

# check for no input arguments and show help
if [ $# -eq 0 ];
then
    help
    exit 1
fi

# parse input arguments
while [ $# -gt 0 ] ; do
    case $1 in
        -h | --help)
            help
            exit 0
            ;;
        -d | --glass-dir)
            GLASS_DIR=$2
            shift 2
            continue
            ;;
        -s | --start-ref)
            START_REF=$2
            shift 2
            continue
            ;;
        -e | --end-ref)
            END_REF=$2
            shift 2
            continue
            ;;
        -a | --account)
            ACCOUNT=$2
            shift 2
            continue
            ;;
        -g | --gpu)
            CPU_OR_GPU="gpu"
            shift 1
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

if [[ "$START_REF" == "" || "$END_REF" == "" ]]
then
  echo "START_REF and END_REF must be provided"
  help
  exit 1
fi

if [[ "$GLASS_DIR" == "" ]]
then
  echo "GLASS_DIR must be provided"
  help
  exit 1
fi

if [[ "$ACCOUNT" == "" ]]
then
  echo "ACCOUNT must be provided"
  help
  exit 1
fi

BENCHMARKS_DIR="$GLASS_DIR/tests/benchmarks"

# Load python modules
module load PrgEnv-gnu cray-python

# Setup base environment
rm -rf "${GLASS_DIR:?}/$START_VENV" # Cleanup old venv
python -m venv --system-site-packages "$GLASS_DIR/$START_VENV"
source "$GLASS_DIR/$START_VENV/bin/activate"
if [ $CPU_OR_GPU = "gpu" ]; then
  python -m pip install --group test --group archer2-gpu
else
  python -m pip install --active --group test
fi
python -m pip uninstall glass -y # Make sure no installation of glass already exists
python -m pip install "git+$GLASS_REPO_URL@$START_REF"

# Setup head environment
rm -rf "${GLASS_DIR:?}/$END_VENV" # Cleanup old venv
python -m venv --system-site-packages "$GLASS_DIR/$END_VENV"
source "$GLASS_DIR/$END_VENV/bin/activate"
if [ $CPU_OR_GPU = "gpu" ]; then
  python -m pip install --active --group test --group archer2-gpu
else
  python -m pip install --active --group test
fi
python -m pip uninstall glass -y # Make sure no installation of glass already exists
python -m pip install "git+$GLASS_REPO_URL@$END_REF"

# Remove old benchmark results
rm -rf "$BENCHMARKS_DIR/outputs"

# Submit job
sbatch --account="$ACCOUNT" "$BENCHMARKS_DIR/archer2/submission_script_$CPU_OR_GPU.sh" "$GLASS_DIR"
