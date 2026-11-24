# V11 - Lightweight long shaft
Actual flat-ended cylinder, nominal 220 mm x 26 mm, 55 g. Effective uniform density 470.873 kg/m3; half the V10 density. Actual mass and inertia recomputed for each randomized cylinder. This is an idealized homogeneous rigid body, not a selected material or a manufactured component.

V10 mount retained all 24/24 trials at 55, 45, and 35 g on the long shaft, but none passed opposition. Re-searching the thumb mount and tendon loads at 55 g (1,200 candidates, 118 nominal successes) produced a new opposed grasp. The selected mount passed calibration 8/8, then 32/32 holdout.

Independent further validation: mass calibration 24/24 at 55 g; nominal perturbations 64/64 opposed-grasp successes; full perturbations 63/64. All 128 blind trials retained the free rod. Maximum sampled self-contact penetration was zero; maximum object-contact penetration was about 1.0001 mm. These are sampled soft-contact measurements, not a guarantee of no penetration at every integration step. No rotation PPO is claimed for V11.

- mount-v10-reference.json: prior mount for comparison.
- runs/v10-mount-mass-ablation.json: failed opposition with the old mount.
- mount.json: selected new mount.
- runs/mount-search.json, mount-calibration.json, mount-holdout.json: search and validation evidence.
- runs/validation.json: independent randomized holdout.
- runs/preview-audit.json: exact-geometry replay audit.

Reproduce with the repository's tested Python dependencies:
    python search_mount.py --n 1200 --workers 12
    python validate_longshaft.py
    python export_preview.py
    node /path/to/teardown/sims/build.mjs ./handuse-longshaft-preview
    node test_preview.mjs
The preview uses recorded MuJoCo transforms, pale-blue gridded surface, flat caps, and a wider camera view. Grid is visual only.
