#!/bin/bash --login

# Load GPU modules
module load PrgEnv-amd/8.6.0
module load rocm
module load craype-accel-amd-gfx90a
module load craype-x86-milan

# Ensure the rocm library and build is known to jax
export LD_LIBRARY_PATH="/opt/rocm/lib:$LD_LIBRARY_PATH"
export ROCM_PATH="/opt/rocm"
