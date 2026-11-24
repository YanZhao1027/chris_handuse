# V14 - Shadow E3M5 large-angle, two-plane manipulation and contact transitions

V13 completed 600 updates (157,286,400 policy interactions; 1,572,864,000 physical integration steps). Its selected update 575 achieved 96/96 on a fresh +/-15 degree CPU test, with mean directed-axis error 1.625 degrees, but only 5/96 on +/-30 degrees. All 288 blind scenarios retained the rod. The adaptive V13 gate never expanded the training range beyond 9-15 degrees.

V14 keeps the official E3M5 model, the 220 x 26 mm 55 g shaft, fixed wrist targets, and the V13 256-state free-grasp reset bank. It initializes the actor and critic from V13 best.pt, lowers exploration noise and creates a fresh optimizer. No new collision exclusions or object forces. No Adroit weight transfer.

Changes:
- Full legal actuator range: original linear residual mapping exactly preserved for |normalized action| <= 0.75; beyond that, interpolate to the respective actuator limit. CPU/GPU transform endpoint and parity tests pass. Thus the warm start is not mathematically identical to V13 outside the inner region.
- 300 control steps (6 s), 2 ms substeps, 50 Hz control. The normalized phase feature is rescaled to the longer horizon.
- Explicit curriculum with 25% original planar +/-15-degree anchors. Planar frontier grows from 20 to 45 degrees over the first 75% of training, without requiring the previous stochastic success gate. Secondary plane (rotation about palm Z, rather than palm Y) gradually introduced after 20% of training, with up to 25% of episodes by 70%. Secondary-plane magnitudes start at 8 degrees and extend up to 30 degrees. The goal remains directed shaft axis, not complete SO(3) orientation or axial spin.
- Recontact telemetry: a previously loaded digit is below 0.05 N for five consecutive control samples (0.1 s), then above 0.10 N for five consecutive samples. Count completed events and successful episodes containing them. This is evidence of a contact transition, not necessarily a full grasp relocation. Do not reward gratuitous contact loss. Reward still encourages opposition and safe retention while allowing temporary unloading.
- Unchanged strict success gate: final 20 consecutive steps within 5 degrees, angular speed below 1.4 rad/s, >=75% opposed samples, retained free rod, drift <40 mm, peak contact penetration below 1.5/3 mm. Peak contact depth audited each integration step.
- 256-world GPU smoke test (inherited policy, mixed plane/angle targets before final anchor sampling adjustment): 256/256 retained, 117 successful, 2 recontacts, no nonfinite states, peak self/object depth 0.155/0.722 mm. Eight CPU smoke trials retained, and both original +/-15-degree cases passed.

## Training
8192 worlds, 1024 physical variants, 900 updates x 32-step rollout = 235,929,600 planned policy interactions / 2,359,296,000 planned integration steps. Fresh CPU validation every 50 updates on 48 cases (4 per signed plane/angle group). Two checkpoint tracks prioritize preservation of +/-15-degree skills then balanced larger angles and mixed-plane success. Independent blind tests cover +/-15, +/-30, +/-45, secondary-plane +/-15/25, and stronger perturbation. The comparator uses V13 weights under V14 controls/horizon, not the original V13 environment. Success-containing recontact trajectories are preferred for example replays. No new learned result claimed until tests finish.

Run:
    python smoke.py
    python train.py --worlds 8192 --variants 1024 --rollout 32 --updates 900 --val-n 48 --eval-every 50 --out runs/ppo01 --warm ../shadow-v13/runs/ppo01/best.pt
    python assess.py
Progress: python progress.py
Job: 2906cc43be23b00d (bounded, stops on detected nonfinite state/capacity overflow).

V13 replay from actual held-out trajectories, first two +/-15-degree cases:
    MUJOCO_GL=egl python render_replay.py ../shadow-v13/runs/ppo01/replay--15.json ../shadow-v13/runs/ppo01/replay-+15.json --out ../shadow-v13/runs/ppo01/preview.mp4
Rendering-only model adds a grid material and lighting without changing recorded dynamics. Videos are local and excluded from Git.
