# TENDON / 06 — elongated-rod in-hand rotation

An 80.3 x 36.2 mm rounded rod (prolate ellipsoid, 55 g nominal) is placed transversely across the five-finger hand. The long axis is local X. The target is a -15 degree world-Z rotation relative to the rod's actual orientation at the start of manipulation. This avoids confusing initial grasp settling with rotation.

## Geometry and GPU collision
- rod_shape.py: rod geometry, mass and uniform-ellipsoid inertia.
- MuJoCo geom_aabb uses center + half-size, not min/max. The corrected GPU model uses [0,0,0,half_length,radius,radius]. The first prototype incorrectly supplied min/max and caused missed GPU broadphase contacts; that model is superseded.
- gpu_rod_base.py: MuJoCo Warp batched GPU model and contact reduction.
- manipulate_rod.py: frozen V4 grasp prior, 10-axis differential residual control, quaternion target tracking and stability reward.
- CPU/GPU same-seed parity checked at step resolution after AABB correction (parity_rod.py, parity_zero.py).

## Search and validation
- action_search_rod.py: 1,024 candidate differential actions x 8 replicates = 8,192 GPU worlds per search.
- Target -15 degrees, scale x2: best candidate #861, action [0,0,0,0,0,0.5,0.75,0,0,-0.25].
- This corresponds to residual yaw commands of approximately [+16,+24,0,0,-13] degrees relative to the V4 grasp controller, subject to joint limits.
- GPU search sample: 62.5% target success, 100% retained for candidate #861.
- Independent CPU reference blind set: 96 new seeds, 49/96 (51.0%) target success, 96/96 retained, no nonfinite states. Mean target error 7.22 degrees.
- Paired demonstration set: 48 new seeds; fixed hold-only control 0/48 target success and 30/48 retained; differential control 29/48 target success and 47/48 retained.
- Representative pair, seed 886000: hold-only final target error 18.91 degrees; differential control final error 3.32 degrees and 98 consecutive 25 Hz steps within the success band. Relative rotation from the start of manipulation is 15.20 degrees around axis [-0.218,-0.019,-0.976], with a -14.83 degree component about world Z.
- The controller is a GPU-searched fixed differential command, not a trained PPO rotation policy. It is direction-specific; the positive +15 degree direction did not find a comparably robust action in this search. This is an early manipulation prototype, not a general-purpose dexterous policy.

## Replay
Open the offline replay with:
xdg-open ~/teardown-demo/tendonhand/rod-v6/teardown-out/dist/tendonhand-rod-live.html

The page is an offline Three.js replay of real CPU MuJoCo trajectories, with a hold-only vs rotate comparison, timeline, contact markers, joint readouts, and measured quaternion rotation.

## Reproduce
PY="$HOME/rm-isaaclab3/env_isaaclab/bin/python"
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" action_search_rod.py --scale 2 --sign -1 --angle 15
OPENBLAS_NUM_THREADS=1 OMP_NUM_THREADS=1 "$PY" eval_15.py
node "$HOME/.agents/skills/teardown/sims/build.mjs" ./tendonhand-rod-live
node test_browser.mjs
