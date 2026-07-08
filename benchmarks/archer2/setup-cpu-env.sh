#!/bin/bash --login

uv venv --clear --python 3.12
uv sync --group benchmarks
