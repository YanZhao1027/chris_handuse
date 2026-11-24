import json,sys
from pathlib import Path
p=Path(sys.argv[1] if len(sys.argv)>1 else 'runs/ppo01')
h=p/'history.json'
if not h.exists():print('Waiting for first update');raise SystemExit
rows=json.loads(h.read_text());r=rows[-1]
print('TENDON / 03 — PPO')
print(f"Update {r['update']} | steps {r['steps']:,} | curriculum {r['difficulty']:.2f} | elapsed {r['seconds']:.1f}s")
print('Validation (success / mean episode return)')
for v in rows:
    if 'validation' not in v:continue
    s=v['validation']['success'];ret=v['validation']['meanReturn']
    print(f"{v['update']:4d}  {'#'*round(s*30):30s}  {s:6.1%}  {ret:7.2f}")
t=p/'test_results.json'
if t.exists():
    print('\nHeld-out paired trials')
    for v in json.loads(t.read_text())['results']:
        print(f"Difficulty {v['difficulty']:.1f}, {v['seconds']:.1f}s: fixed {v['baseline']['success']:.1%}  PPO {v['ppo']['success']:.1%}  N={v['ppo']['n']}")
