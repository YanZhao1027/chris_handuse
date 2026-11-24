# TENDON / 02 — rigid-body grasp lab
V1 is preserved in the parent folder. This is an original illustrative hand with 20 hinge coordinates, 5 fixed (constant moment arm) tendons, 5 yaw position servos and spring returns. MuJoCo 3.8.1 solves dynamics, unilateral contacts, friction and gravity. Wrist movement is prescribed via mocap. The free sphere is initially located by a weld which is disabled at t=1 s, before the wrist lift (65 mm). No object support remains. Finger-to-finger/self-collision is omitted; geometry/material parameters are illustrative, not a manufactured design. No real motor model or physical experiment calibration. Joint limits are soft constraints.
## Quick start
Use the existing ~/rm-isaaclab3/env_isaaclab/bin/python.
Open http://127.0.0.1:18765/ while the local solver is running.
Offline replay: teardown-out/dist/tendonhand-physics.html (no server necessary; custom trials require server).
From this directory:
```sh
PY="$HOME/rm-isaaclab3/env_isaaclab/bin/python"
"$PY" validate.py
"$PY" pack.py
node "$HOME/.agents/skills/teardown/sims/build.mjs" ./tendonhand-physics
"$PY" server.py
```
Run browser tests in another terminal: node test-browser.mjs.
## Files
- model.py / results/hand.xml: MJCF model
- trial.py: deterministic integration, contact force extraction and lift protocol
- search.py / results/search.json: parameter search (1500 second-stage runs)
- validate.py / results/validation.json: 6-second nominal, no-pull, release and 12 sensitivity cases
- results/grasp.json / zero-force.json / release.json: complete 100 Hz sampled trajectories
- tendonhand-physics: Three.js playback using solved state, 50 Hz samples and actual contacts
- results/browser-grasp.png / browser-zero.png / browser-custom.png: Playwright output
- results/native-open.png: NVIDIA EGL MuJoCo render
- server.py: local-only validated simulation endpoint; no external access
