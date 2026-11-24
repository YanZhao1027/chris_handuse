# TENDON / 03 — residual PPO
Uses existing V2 MJCF and ~/rm-isaaclab3/env_isaaclab/bin/python; no dependency changes. CPU MuJoCo physics, 1 ms integration, 25 Hz closed-loop control. Five independent tendon-force residuals and thumb yaw residual around V2 nominal. Residuals bounded and low-pass filtered. Policy receives normalized proprioception, object-relative position/velocity and initial location, five contact forces and tendon lengths, previous action, task phase, radius. Mass and friction are randomized but not observed. Wrist lift prescribed. Initial weld disabled at 1 s. No hand self-collision, no hardware/calibrated force model. Episode success: after 2 s, at least two loaded digits, drift <25 mm, sphere lifted >35 mm above initial throughout hold, fixture off and episode completed. Early drops terminate.
Train curriculum difficulty .20 -> 1.0; randomizes object center, radius, mass, friction. 48 environments in 6 CPU workers; PPO tanh Gaussian actor, 2x96 MLP, GAE, clipped policy/value loss, entropy. Validation seeds 100000+, checkpoint selected by success then reward. Test seeds 20270000+, untouched during training, at 0.3/1.0/1.5 difficulty and 4/6 s durations. No claim of real-world generalization.
```sh
PY="$HOME/rm-isaaclab3/env_isaaclab/bin/python"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" train.py --updates 120 --envs 48 --workers 6 --rollout 64 --eval-every 5 --val-n 32 --out runs/ppo01
"$PY" assess.py --run runs/ppo01 --n 96
```
Checkpoints: initial.pt, best.pt, latest.pt. Resume with --resume runs/ppo01/latest.pt and an increased total --updates. History JSON includes training and validation stats. Test results include each paired trial. Original V1/V2 and their local server are untouched.
