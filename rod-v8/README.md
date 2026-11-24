# TENDON / 08 - Bidirectional flat-shaft manipulation
This experiment uses a 110 x 26 mm cylinder with flat end faces, light-blue rendering and a procedural grid texture. The texture is visual only; friction remains a randomized contact coefficient. The old V7 capsule experiment is unchanged.

Protocol: freeze the V4 grasp controller. Disable initial positioning weld at 1 s, lift 65 mm, set a relative world-Z target at 2 s, then execute differential finger commands until 6 s. No direct object torque or pose reset after release. Success requires full-quaternion error <8 degrees for the final 0.4 s, angular speed <1.2 rad/s, fixture off and continued stable retention.

- search_bidir.py: 512 candidate actions x 16 shared physics variants, with both signs scored on the same motions. Independent 64-variant calibration. At +/-15 degrees best fixed-action candidates scored 47/64 and 63/64 respectively, both 64/64 retained. +/-30 degrees are not yet reliably controllable.
- bc.py: collect physically successful demonstrations with independent parameters, balance the available positive and negative samples, and train an observation-dependent action network initialized with searched-action priors.
- train_v8.py: 8192 worlds x 32 steps x 120 PPO updates = 31,457,280 planned transitions. Ten continuous residual actions. Independent CPU validation chooses checkpoint primarily by worse-direction success. The final blind set is separate.
- assess_v8.py: independent CPU evaluation of hold-only, searched-action, BC, and PPO controllers, including wider randomization and +/-30 degrees extrapolation. Representative paired trajectories for both directions are exported.
- export_replay.py: build actual recorded physical trajectories into the blue, flat-ended, gridded replay. No synthetic rotation.

Progress: python3 progress.py
Job 2cbea776038901ec, operation tendon08-bc-ppo-20261124.
Checkpoint paths: runs/bc.pt, runs/bidir01/best.pt, runs/bidir01/latest.pt.
After the job finishes, replay: teardown-out/dist/tendonhand-rod-v8-live.html.
