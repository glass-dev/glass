#!/bin/bash -l

# Request ten minutes of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=3:0:0

# Request 1 gigabyte of RAM (must be an integer followed by M, G, or T)
#$ -l mem=4G

# Request exclusive access to a node
#$ -ac exclusive=true

# Set the name of the job.
#$ -N glass_regression_test

# Set the working directory to somewhere in your scratch space.
#  This is a necessary step as compute nodes cannot write to $HOME.
# Replace "<your_UCL_id>" with your UCL user ID.
#$ -wd /home/ccaecai/Scratch/glass

GLASS_DIR="$HOME/Scratch/glass"
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

# Set the base and head refs to be compared
BASE_REF="<some-git-commit-hash>"
HEAD_REF="<some-git-commit-hash>"

# Load modules and pre-installed python dependencies through venv
# shellcheck source=/dev/null
source "$BENCHMARKS_DIR/myriad/load_modules.sh"
# shellcheck source=/dev/null
source "$GLASS_DIR/.venv/bin/activate"

# Remove old benchmark results
rm -rf "$BENCHMARKS_DIR/outputs"

# Make sure to installation of glass already exists
python -m pip uninstall glass -y

# Install the base ref of glass
python -m pip install "git+$GLASS_REPO_URL@$BASE_REF"

# Generate the base report for comparison later
python -m pytest "$BENCHMARKS_DIR" --benchmark-autosave "${BENCHMARKS_SHARED_FLAGS[@]}"

# Uninstall the base ref of glass
python -m pip uninstall glass -y

# Install the head ref of glass
python -m pip install "git+$GLASS_REPO_URL@$HEAD_REF"

# Run the stable and unstable benchmarks and compare to the base ref
python -m pytest "$BENCHMARKS_DIR" -m stable --benchmark-compare=0001 --benchmark-compare-fail=mean:5% "${BENCHMARKS_SHARED_FLAGS[@]}"
python -m pytest "$BENCHMARKS_DIR" -m unstable --benchmark-compare=0001 --benchmark-compare-fail=mean:0.0005 "${BENCHMARKS_SHARED_FLAGS[@]}"
