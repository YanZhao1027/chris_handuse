"""Read the latest atomic on-disk training report (no GPU allocation)."""
import json,sys
from pathlib import Path
root=Path(__file__).parent
p=root/(sys.argv[1] if len(sys.argv)>1 else 'runs/ppo-gpu01')
for n in (512,2048,4096,8192):
 f=root/f'runs/bench-{n}-80.json'
 if f.exists():
  b=json.loads(f.read_text());print(f"{n:5,} worlds | {b['policy_steps_per_second']:10,.0f} transitions/s | {b['physics_steps_per_second']/1e6:.3f}M physics steps/s")
h=p/'history.json'
if not h.exists():
 print('No training update yet. Check AI Commander job logs.')
 sys.exit(0)
hist=json.loads(h.read_text());last=hist[-1];cfg=json.loads((p/'config.json').read_text())
target=cfg['worlds']*cfg['rollout']*cfg['updates']
print(f"\nUpdate {last['update']} / {cfg['updates']} | {last['steps']:,} / {target:,} ({last['steps']/target:.1%})")
print(f"Throughput {last['stepsPerSecond']:,.0f} transitions/s; total physics steps {last['physicsSteps']:,}; free VRAM {last['vramFreeGiB']:.2f} GiB")
if last['trainSuccess'] is not None:print(f"Completed episode success (training) {last['trainSuccess']:.1%} across {last['episodes']:,} episodes")
print('CPU reference validation:')
for row in hist:
 if 'validation' in row:
  s=row['validation']['success'];print(f"  {row['update']:4d}  {'#'*round(s*40):40s}  {s:.1%}")
r=p/'test_results.json'
if r.exists():
 print('New blind test:')
 for v in json.loads(r.read_text())['results']:
  print(f"  difficulty {v['difficulty']}: fixed {v['fixed']['success']:.1%} | CPU PPO {v['cpuPPO']['success']:.1%} | GPU PPO {v['gpuPPO']['success']:.1%}")
