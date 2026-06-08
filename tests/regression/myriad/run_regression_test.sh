#!/bin/bash -l

# Request three hours of wallclock time (format hours:minutes:seconds).
#$ -l h_rt=3:0:0

# Request 4 gigabyte of RAM (must be an integer followed by M, G, or T)
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
REGRESSION_DIR="$GLASS_DIR/tests/regression"
REGRESSION_SHARED_FLAGS=(
  "--benchmark-storage=file://$REGRESSION_DIR/outputs"
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
source "$REGRESSION_DIR/myriad/load_modules.sh"
# shellcheck source=/dev/null
source "$GLASS_DIR/.venv/bin/activate"

# Remove old regression test results
rm -rf "$REGRESSION_DIR/outputs"

# Make sure to installation of GLASS already exists
python -m pip uninstall glass -y

# Install the base ref of GLASS
python -m pip install "git+$GLASS_REPO_URL@$BASE_REF"

# Generate the base report for comparison later
python -m pytest "$REGRESSION_DIR" --benchmark-autosave "${REGRESSION_SHARED_FLAGS[@]}"

# Uninstall the base ref of GLASS
python -m pip uninstall glass -y

# Install the head ref of GLASS
python -m pip install "git+$GLASS_REPO_URL@$HEAD_REF"

# Run the stable and unstable regression tests and compare to the base ref
python -m pytest "$REGRESSION_DIR" -m stable --benchmark-compare=0001 --benchmark-compare-fail=mean:5% "${REGRESSION_SHARED_FLAGS[@]}"
python -m pytest "$REGRESSION_DIR" -m unstable --benchmark-compare=0001 --benchmark-compare-fail=mean:0.0005 "${REGRESSION_SHARED_FLAGS[@]}"
