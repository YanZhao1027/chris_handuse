# TENDON / 04 — GPU parallel RL

V4 preserves V1, V2, V3 and the V2 local server. It reuses the installed MuJoCo Warp 3.8.1, Warp 1.13.0 and PyTorch 2.10 CUDA environment. All physical worlds are batched on the RTX 5090 D v2. CUDA graphs drive the 1 ms solver and wrist/yaw/tendon schedules. Zero-copy Torch/Warp arrays carry state, actions and contact forces. PPO inference, returns, GAE and gradient updates run on CUDA. GPU arrays are reset by a world mask. Only scalar diagnostics and CPU validation transfer to host. Initial positioning weld is disabled at 1 second, before the wrist lifts. V3 residual action limits, rewards, success criteria and policy architecture preserved. No self collision or calibrated hardware model yet.

## Validation
- probe.py: compile/capture on 32 worlds.
- bench.py --worlds 96 --steps 100 --exact: CPU and GPU same domain seeds and fixed controller; 96/96 binary outcome agreement (24 successes each). This is not a claim of bit-identical trajectories.
- bench.py --worlds 512/2048/4096/8192 --steps 110: dynamics throughput, finite state, contact/constraint buffers. JSON in runs/.
- Randomized physical constants are precomputed by CPU MuJoCo, then batched on GPU. A pool of 1024 mass/radius/friction variants is used for training; physical values remain fixed within a world, object position is resampled on reset.
- Warm start from V3 best.pt. 32 CPU validation seeds are the same as V3; selection does not use new blind seeds. assess_gpu.py tests fixed control, V3 CPU PPO and V4 GPU PPO on new seeds 40270000+ (96 paired trials per difficulty).

## Run
```sh
cd ~/teardown-demo/tendonhand/gpu-v4
PY="$HOME/rm-isaaclab3/env_isaaclab/bin/python"
"$PY" progress.py
```
Training job b2a96a3c4d791579 benchmarks 8192 worlds; if successful, launches 8192 worlds x 32 steps x 120 updates (31,457,280 transitions), otherwise 4096. Checkpoints saved under runs/ppo-gpu01, including initial.pt, best.pt, latest.pt (optimizer and history), config.json and history.json. A new training process can resume with --resume runs/ppo-gpu01/latest.pt and matching args. No automatic restart; completion launches blind evaluation within the same job.
