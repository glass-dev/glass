#!/bin/bash -l

GLASS_DIR=""
GLASS_REPO_URL="https://github.com/glass-dev/glass"
START_REF=""
END_REF=""
START_VENV=".venv-start"
END_VENV=".venv-end"
ACCOUNT=""
SETUP_ENVS="true"

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
  echo "    --skip-setup                     Flag to state if the setup (installation of "
  echo "                                     dependencies) can be skipped. Good for simply"
  echo "                                     re-submitting"
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
        --skip-setup)
            SETUP_ENVS="false"
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


if [[ "$SETUP_ENVS" == "true" ]]
then
  # Setup base environment
  rm -rf "${GLASS_DIR:?}/$START_VENV" # Cleanup old venv
  uv venv  "$GLASS_DIR/$START_VENV"
  source "$GLASS_DIR/$START_VENV/bin/activate"
  uv sync --active --group test
  uv pip uninstall glass -y # Make sure no installation of glass already exists
  uv pip install "git+$GLASS_REPO_URL@$START_REF"

  # Setup head environment
  rm -rf "${GLASS_DIR:?}/$END_VENV" # Cleanup old venv
  uv venv "$GLASS_DIR/$END_VENV"
  source "$GLASS_DIR/$END_VENV/bin/activate"
  uv sync --active --group test
  uv pip uninstall glass -y # Make sure no installation of glass already exists
  uv pip install "git+$GLASS_REPO_URL@$END_REF"

  # Remove old benchmark results
  rm -rf "$BENCHMARKS_DIR/outputs"
fi

# Submit job
sbatch --account="$ACCOUNT" "$BENCHMARKS_DIR/archer2/submission_script_cpu.sh" "$GLASS_DIR"
