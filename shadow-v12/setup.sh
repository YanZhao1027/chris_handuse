#!/bin/sh
set -eu
cd "$(dirname "$0")"
PY="${PYTHON:-$HOME/rm-isaaclab3/env_isaaclab/bin/python}"
mkdir -p .deps runs
"$PY" -m pip install --index-url https://pypi.org/simple --no-deps --target "$PWD/.deps" gymnasium-robotics==1.4.1 minari==0.5.3 pettingzoo typer rich click shellingham markdown-it-py mdurl
# Removed in newer MuJoCo; no dynamics option is changed.
sed -i 's/ apirate="200"//' .deps/gymnasium_robotics/envs/assets/adroit_hand/adroit_assets.xml
