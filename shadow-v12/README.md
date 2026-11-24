# V12 — Adroit Pen baseline and Shadow E3M5 assets

Two deliberately separate robot models:
- Standard AdroitHandPen-v1, the environment used by the Minari D4RL/pen/expert-v2 dataset: 45 observations, 24 actions, 2 ms physics, 5 substeps per control, 200-step online benchmark.
- Menagerie Shadow Hand E3M5: actual upstream 24-joint / 20-actuator model with four coupling tendons. Pinned commit eea4c7431b63da8278b0ff31638e0b0e941cec0e; fetch_shadow.py downloads the shadow_hand directory only, verifies Git blob SHA and records SHA256. Upstream Apache-2.0 license is downloaded with the model. The original assets are not copied to this repository.

## Compatibility and reproducibility
Use the existing repository simulation environment (Python 3.12, MuJoCo 3.8.1 and CUDA PyTorch). setup.sh installs Gymnasium-Robotics 1.4.1, Minari 0.5.3 and required Python dependencies in an isolated .deps directory, not the system environment. It removes the obsolete apirate XML attribute only; no dynamics, action scaling or reward changes. Standard Adroit uses 20 NoSlip iterations; MuJoCo Warp 3.8.1 explicitly rejects NoSlip. Thus online training and evaluation preserve original CPU MuJoCo dynamics, and network training runs on CUDA. No equivalence to a modified GPU task is assumed.

From this directory:
    sh setup.sh
    . ./env.sh
    "$PY" fetch_shadow.py
    "$PY" -c 'import minari; minari.load_dataset("D4RL/pen/expert-v2", download=True)'
    "$PY" train_bc.py --epochs 30 --out runs/bc01
    "$PY" train_ppo.py --workers 12 --per 8 --rollout 64 --updates 100 --out runs/ppo01
    "$PY" shadow_probe.py
    "$PY" render_preview.py

Raw expert data is downloaded locally from Farama/Minari and is not redistributed. No raw data, third-party package files, downloaded models, local paths containing credentials, or videos are committed. See https://minari.farama.org/datasets/D4RL/pen/expert-v2/ and https://github.com/google-deepmind/mujoco_menagerie/tree/main/shadow_hand.

## Actual results
Dataset: 4,958 trajectories / 499,206 transitions. Episode-disjoint split: 4,462 train episodes (449,371 transitions), 496 held-out episodes (49,835 transitions). BC uses train-only observation statistics and 30 epochs. Checkpoint selection uses distinct online CPU validation seeds. Success criterion used in this project: environment success for at least 20 consecutive 100 Hz steps (0.2 seconds); final success is separately reported. The stored dataset and 200-step online benchmark have different episode lengths, so do not compare their returns as if identical.

BC blind set (96 independent seeds): zero 4/96 stable; random 0/96 stable; BC 79/96 stable, 78/96 final, 1/96 ever below 75 mm.

PPO: 96 CPU environments in 12 worker processes; CUDA actor/critic; 100 updates / 614,400 policy interactions / 3,072,000 physics steps. Initialized from BC, retains a small BC anchor, and uses independent validation seeds. The 200-step time limit is treated as a finite-horizon terminal for GAE. Both BC and selected PPO evaluated on a new shared 96-seed blind set: BC 75/96 stable, PPO 81/96 stable. PPO final success 81/96; BC 72/96. PPO mean return 8549.55 vs BC 7664.63. PPO best checkpoint update 100 (check runs/ppo01/blind.json). This is modest improvement on the paired sample, not proof of a universal advantage or hardware transfer.

Menagerie original E3M5 scene passed 500 CPU physics steps with finite state; see runs/shadow-cpu-probe.json for exact named actuator mapping and dimensions. GPU compatibility probe is a separate experiment. Neither Adroit BC nor PPO has been transferred to E3M5 or the 220 mm rod. Next step is task geometry and control retargeting, independent grasp validation, then new rotation training.

Local replay: runs/bc01/preview.mp4 is rendered by MuJoCo itself from a representative held-out BC trajectory (seed 1203000); 100 frames at 25 fps, two-times slow motion. It is not a population estimate.
