# Source from shadow-v12 directory.
export PYTHONPATH="$PWD/.deps${PYTHONPATH:+:$PYTHONPATH}"
export MINARI_DATASETS_PATH="$PWD/.datasets"
export HF_HUB_DISABLE_IMPLICIT_TOKEN=1
export OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1
PY="${PYTHON:-$HOME/rm-isaaclab3/env_isaaclab/bin/python}"
