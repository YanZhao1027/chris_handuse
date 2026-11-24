# TENDON / 09 - Contact-aware in-hand manipulation
V8 had disabled finger-finger collision. Actual recorded V8 poses show up to approximately 15 mm overlap between thumb and palm and 13 mm between thumb and middle finger. V9 preserves V8 for comparison and changes the physical model, not just its appearance.

- Thumb mount relocated from (-40,-21,0) mm to (-49,-21,8) mm in palm frame.
- Inter-digit collisions, nonadjacent links on each digit, and palm/drive collisions with all distal links and all thumb links enabled. Adjacent hinge-connected link overlap and four proximal link/palm mounting interfaces intentionally excluded. Mount hardware itself remains an idealized joint.
- Stiffer soft-contact solver. Finger/object and finger/finger contacts solved every 1 ms. Contact penetration measured at policy samples (40 ms); success includes <1 mm self-contact and <2 mm object-contact peak at sampled steps.
- Independent CPU geometry audit uses mj_geomDistance on intended non-connected collision pairs. Initial 8 scene audit: 8/8 retained, worst sampled self overlap 0.016 mm; old commands did not achieve rotation goals. GPU 256-scene hold probe: 256/256 retained, zero nonfinite states.
- Flat-ended light-blue 110x26 mm cylinder and grid visual texture retained. Renderer uses revised thumb mount and actual capsule dimensions; decorative hinge pins removed. Grid is visual only.
- Preview audit (runs/preview-audit.json): four actual CPU trajectories; highest sampled geometric self penetration 0.115 mm; rod-contact geometry peak ~0.764 mm. Includes an unsuccessful +15 candidate. This is not a trained policy.
- Search: 4096 GPU worlds; independent 64-variant calibration. Static -15 best 64/64; +15 best only 3/64 with poor retention. search_sequence.py tests two-phase differential actions including release pulses, then independent calibration.
- BC only successful trajectories; balances both signs where available. Residual command below -0.85 enables full tendon release, while prior commands remain unchanged. PPO curriculum targets signed 12-30 degrees; starts <=15 degrees and increases to 30 degrees; 4096 worlds x 32 steps x 160 updates = 20,971,520 planned policy transitions. Self-contact constraints and sampled penetration criteria are part of reward and success.
- CPU validation signed +/-15,20,25,30 degrees, 64 independent scenarios, checkpoint selection by weaker sign. Independent blind tests include nominal and wider randomization, then +/-20,25,30. No claim that regrasp or larger-angle tasks have been learned until evaluated.
- Job: 57c6a82df94ac2db. Search -> successful demonstrations -> BC -> PPO -> blind test -> replay.
Progress: python3 progress.py
Preview: teardown-out/dist/tendonhand-v9-preview.html
Final replay (after successful pipeline): teardown-out/dist/tendonhand-rod-v9-live.html
