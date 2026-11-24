# TENDON / 01 — Teardown robotic hand
Original illustrative five-finger design, not a CAD replica or a manufacturable hand. Reference concepts: Shadow Dexterous Hand, https://shadowrobot.com/dexterous-hand-series/ ; tendon-driven finger/pulley research, https://pubmed.ncbi.nlm.nih.gov/33501063/ . All dimensions and elastic constants in kin.js are illustrative. 20 modeled joint coordinates (4 per digit), coupled by five curl inputs plus global spread/opposition. No independent 20-axis control, collision solver, contact physics, or measured force. A visual tendon routing and constant moment-arm calculation are idealizations. Tip gap is center-to-center, not surface clearance.
## Build and test
```sh
cd ~/teardown-demo/tendonhand
D="$HOME/.agents/skills/teardown"
node "$D/sims/build.mjs" ./tendonhand-live ./tendonhand-film
node --test tests/kin.test.js
node tests/browser.mjs
xdg-open teardown-out/dist/tendonhand-live.html
```
## Capture
```sh
node "$D/sims/capture.mjs" tendonhand-film --size 1600x900 --dpr 1 --stills 1.8,6.2,10.1,14.1,18.1,22.1 --cols 2 --out teardown-out/review/film-sheet.png
node "$D/sims/capture.mjs" tendonhand-film --size 1600x900 --dpr 1 --fps 60 --out teardown-out/tendonhand.mp4
```
Live mode: presets, sliders, isolate, view modes, orbit, auto tour; manual control resumes auto tour after 25s inactivity. S is finger spread; Y is thumb basal yaw. Film is a deterministic 25-second 60 fps timeline. The HTML runs offline. No system changes or original Teardown example edits.
