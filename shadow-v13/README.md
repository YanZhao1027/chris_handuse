# V13 — Shadow E3M5 / 220 mm shaft, large-scale in-hand rotation

Uses the upstream pinned Menagerie Shadow E3M5 assets downloaded by ../shadow-v12/fetch_shadow.py. Original joint topology, tendon couplings, collision geometry and position actuators are retained. Forearm is fixed and reoriented to palm-down with clearance above the floor. A free 220 mm x 26 mm, 55 g cylinder replaces the original ellipsoid. Its long axis is local Z. Two wrist servo targets are held fixed; the policy commands the other 18 actuators, within upstream control limits. No Adroit policy weights are used because the models and controls differ.

## Grasp initialization
4,000 nominal CPU candidates, 327 passed. Best candidate #1811 passed 8/8 calibration and 96/96 independent pose-perturbed grasp trials. Object pose is held during the first 0.7 s of the grasp-search approach only (approach penetration tracked separately), then all pose overwrites stop; scored hold from 1.0 to 1.7 s. No weld or position fixture exists in the training model. This does NOT validate picking the shaft up from a table. The 256 training reset states are generated from separate seeds and are free-body loaded grasps. No validation/blind states enter the training bank.

4,096 GPU worlds held the shaft for 200 control steps / 4 s under independent mass/friction variation. All retained; mean opposed fraction 0.999999, peak contact self-depth 0.0531 mm and object-depth 0.4971 mm. Depths measured at each integration substep using active collision contacts; this is not a proof of geometric nonintersection of every decorative mesh. Original contact exclusions are retained.

## Training
- MuJoCo Warp GPU physics: 2 ms integration, 10 substeps per 50 Hz action, 200-control-step episodes. 1,024 fixed physical variants: mass 44–66 g, sliding friction 0.85–1.30; independent reset poses each episode.
- 90-dimensional state observation, including simulation-provided rod pose, velocity, contact forces and physical parameters. Privileged-state simulation policy, not a deployed sensor-only controller.
- 18 bounded residual servo commands, filtered in time; full control ranges respected. Directed shaft axis in actual palm coordinates, targets rotated about palm Y. Spin about shaft axis is not constrained by this task.
- Initial signed angle goals 9–15 degrees; adaptively expand upper limit in 5-degree increments up to 45 only after both directions exceed 68% rollout success with retention >85% on recent completed episodes.
- Success requires free-body retention, peak palm-relative translation <40 mm, opposing contact for >=75% of samples, final 20 consecutive control steps within 5 degrees, angular speed <1.4 rad/s, and sampled peak self/object penetration below 1.5/3 mm. Opposed contact means radial thumb and ordinary-finger resultant force directions cosine < -0.35, loads >0.1 N, and at least two ordinary digits loaded.
- 8,192 worlds x 32-step rollouts x 600 PPO updates = 157,286,400 planned interactions / 1,572,864,000 integration steps. Checkpoint and CPU validation every 25 updates, plus initial/first/final. Independent CPU validation uses fresh initial pose seeds and independently perturbed physical parameters. The best ±15 degree and broad-angle checkpoints are tracked separately. Final blind tests use separate seeds at ±15, ±30, and wider perturbations including ±45.
- Large bounded run, not an unlimited job. A capacity or nonfinite-state failure stops training. No successful rotation claim until validated.

Job: 634d60e50c1a5553, operation handuse-v13-shadow-ppo-20261124.

## Reproduce
Use Python from the tested simulation environment in ../requirements-simulation.txt. Download the pinned Shadow assets first.
    python search_grasp.py --n 4000 --workers 12
    python probe.py --worlds 4096
    python train.py --worlds 8192 --variants 1024 --rollout 32 --updates 600 --out runs/ppo01
    python assess.py
Progress: python progress.py.
