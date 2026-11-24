# V10 - Opposed thumb / Chris Handuse

V9 solved gross self-intersection but used a thumb that followed the four-finger curl topology. V10 rotates the thumb mount into an opposing flexion plane. The fixed mounting transform maps local +Y towards palm +Z and local curl toward palm +Y, then applies an optimized tilt and twist. The thumb yaw and three flexion joints are retained; there is no new independently driven opposition joint yet.

## Verified design
- 640 CPU candidates: 33 succeeded in the nominal grasp search.
- Independent calibration: best mount succeeded 8/8; the additional untouched holdout succeeded 32/32. Success requires release of the initial weld, lift, continued retention, and opposing loaded side-contact normals for at least 90% of the sampled hold, plus penetration limits.
- Mount parameters in mount.json; each trial uses randomized shaft dimensions, mass, friction, and center. 110 x 26 mm nominal flat cylinder.
- Opposed contact metric: force-weighted contact normals projected perpendicular to the shaft axis, dot-product cosine below -0.35, radial loads over 0.10 N for both thumb and four-finger groups, at least two ordinary digits loaded. The metric does not imply a hardware-validated force closure guarantee.
- Independent GPU/CPU hold probe uses 64 shared seeds, with GPU retention 64/64, opposed fraction 100%, and CPU strict grasp-and-zero-angle success 59/64. The two engines are not bit-identical. Sampled self-contact overlap was zero, not a proof of nonpenetration at every 1 ms substep.
- Exact geometry preview: actual MuJoCo geom_xpos/geom_xmat, exact capsule/box/cylinder sizes, light-blue gridded rod, contrasting thumb. Representative seed 1003000 had zero sampled intended self-collision overlap by mj_geomDistance; shaft overlap 0.750 mm by geometry audit. Grid is a visual marking, not friction geometry.

## Rotation learning (in progress at snapshot)
- New fixed grasp prior; old V4/V8 policies are not used to actuate this redesigned thumb.
- GPU search uses 512 two-phase commands x 8 shared variants and independent 64-replica calibration. No candidate meets the full +/-15 degree rotation criterion in the independent calibration. The BC collection contains zero successful demonstrations, so the saved initialization is a searched-action prior, NOT a learned imitation policy.
- GPU PPO uses 4,096 worlds x 32 steps x 120 updates = 15,728,640 planned policy transitions, signed 12-15 degree goals. Reward and success require real opposing contact; success also requires at least 80% opposed samples after lifting, final 0.4 seconds within 8 degrees and angular-speed limit, continued free-body retention, and sampled self/object penetration limits. Separate CPU validation and blind tests.
- Do not claim bidirectional rotation success until actual independent testing passes.

## Reproduce
Use the environment in ../requirements-simulation.txt with matching CUDA PyTorch. From this directory:
    python search_mount.py --n 640 --workers 12
    python probe.py
    python search_sequence.py
    python bc.py
    python train_v8.py --worlds 4096 --variants 1024 --updates 120 --rollout 32 --val-n 48 --out runs/opposition01
    python assess.py

Preview (real rigid-body grasp, not learned rotation):
    python export_preview.py
    node /path/to/teardown/sims/build.mjs ./handuse-opposition-preview
    xdg-open teardown-out/dist/handuse-opposition-preview.html

Progress: python progress.py.
