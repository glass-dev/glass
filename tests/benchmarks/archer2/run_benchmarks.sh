#!/bin/bash -l

GLASS_DIR=""
GLASS_REPO_URL="https://github.com/glass-dev/glass"
BENCHMARKS_DIR="$GLASS_DIR/tests/benchmarks"
CPU_OR_GPU="cpu|gpu" # <-- Select one

BASE_VENV=".venv-base"
HEAD_VENV=".venv-base"

# Set the base and head refs to be compared
BASE_REF="<some-git-commit-hash>"
HEAD_REF="<some-git-commit-hash>"

# Setup base environment
rm -rf "${GLASS_DIR:?}/$BASE_VENV" # Cleanup old venv
uv venv "$GLASS_DIR/$BASE_VENV"
source "$GLASS_DIR/$BASE_VENV/bin/activate"
if [ $CPU_OR_GPU = "gpu" ]; then
  uv sync --group test --group archer2-gpu
else
  uv sync --group test
fi
uv pip uninstall glass -y # Make sure no installation of glass already exists
uv pip install "git+$GLASS_REPO_URL@$BASE_REF"

# Setup head environment
rm -rf "${GLASS_DIR:?}/$HEAD_VENV" # Cleanup old venv
uv venv "$GLASS_DIR/$HEAD_VENV"
source "$GLASS_DIR/$HEAD_VENV/bin/activate"
if [ $CPU_OR_GPU = "gpu" ]; then
  uv sync --group test --group archer2-gpu
else
  uv sync --group test
fi
uv pip uninstall glass -y # Make sure no installation of glass already exists
uv pip install "git+$GLASS_REPO_URL@$HEAD_REF"

# Remove old benchmark results
rm -rf "$BENCHMARKS_DIR/outputs"

# Submit job
sbatch "submission_script-$CPU_OR_GPU.sh" "$GLASS_DIR"
