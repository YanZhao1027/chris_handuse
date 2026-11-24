"""Create an offline training report from checkpoints and evaluation data."""
import html,json
from pathlib import Path
p=Path('runs/ppo01');history=json.loads((p/'history.json').read_text())
vals=[x for x in history if 'validation' in x]
W,H=880,260;left,top=62,24;right,bottom=W-26,H-39
x=lambda u:left+(right-left)*(u-1)/max(1,120-1)
y=lambda r:bottom-(bottom-top)*r
lines=[]
for i in range(6):
    r=i/5;lines.append(f'<line x1="{left}" y1="{y(r)}" x2="{right}" y2="{y(r)}" stroke="#dce6e5"/><text x="{left-10}" y="{y(r)+4}" text-anchor="end">{r:.0%}</text>')
for u in [1,20,40,60,80,100,120]:lines.append(f'<text x="{x(u)}" y="{bottom+25}" text-anchor="middle">{u}</text>')
baseline=10/32
lines.append(f'<line x1="{left}" y1="{y(baseline)}" x2="{right}" y2="{y(baseline)}" stroke="#6c7e82" stroke-width="2" stroke-dasharray="7 6"/>')
lines.append('<polyline fill="none" stroke="#007e83" stroke-width="3" points="'+' '.join(f"{x(r['update']):.1f},{y(r['validation']['success']):.1f}" for r in vals)+'"/>')
for r in vals:lines.append(f'<circle cx="{x(r["update"])}" cy="{y(r["validation"]["success"])}" r="3.2" fill="#007e83"><title>Update {r["update"]}: {r["validation"]["success"]:.1%}</title></circle>')
test=(p/'test_results.json');rows=[]
if test.exists():
    for r in json.loads(test.read_text())['results']:
        a,b=r['baseline'],r['ppo'];rows.append(f"<tr><td>{r['difficulty']:.1f}</td><td>{r['seconds']:.0f}s</td><td>{a['success']:.1%}</td><td>{b['success']:.1%}</td><td>{b['success']-a['success']:+.1%}</td><td>{b['n']}</td></tr>")
best=max(vals,key=lambda r:(r['validation']['success'],r['validation']['meanReturn']))
latest=history[-1]
doc=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width"><title>TENDON / 03 | PPO report</title><style>
*{{box-sizing:border-box}}body{{margin:0;background:#edf3f1;color:#20383d;font:15px system-ui,sans-serif}}main{{max-width:1040px;margin:auto;padding:32px 24px}}header{{border-bottom:1px solid #bacdca;padding-bottom:22px}}small{{letter-spacing:.13em;color:#007e83;font:600 12px monospace}}h1{{font-size:44px;letter-spacing:-2px;margin:5px 0}}h2{{font-size:19px}}.grid{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:24px 0}}article,section{{background:#fff;border:1px solid #c9d8d5;border-radius:9px;padding:20px}}article span{{display:block;color:#647c80}}article strong{{font:700 29px monospace;display:block;margin-top:10px}}svg{{width:100%;height:auto;font:12px monospace;fill:#5e7378}}table{{width:100%;border-collapse:collapse}}th,td{{text-align:right;padding:12px 9px;border-bottom:1px solid #dce6e5;font-variant-numeric:tabular-nums}}th:first-child,td:first-child{{text-align:left}}p{{line-height:1.5;color:#536b70}}section{{margin-top:18px}}@media(max-width:550px){{.grid{{grid-template-columns:1fr}}main{{padding:16px}}}}
</style></head><body><main><header><small>MECHANISM LAB / LEARNING</small><h1>TENDON / 03</h1><p>Residual PPO · 48 parallel MuJoCo environments · 25 Hz policy</p></header><div class="grid"><article><span>Completed interaction steps</span><strong>{latest['steps']:,}</strong></article><article><span>Best validation success</span><strong>{best['validation']['success']:.1%}</strong></article><article><span>Selected update</span><strong>{best['update']}</strong></article></div><section><h2>Validation learning curve</h2><svg viewBox="0 0 {W} {H}">{''.join(lines)}</svg><p>Teal: deterministic PPO on 32 validation seeds. Dashed: nominal fixed controller (31.25%). Validation selects checkpoint; it is not held-out evidence.</p></section><section><h2>Held-out paired evaluation</h2>{'<table><thead><tr><th>Difficulty</th><th>Duration</th><th>Fixed</th><th>PPO</th><th>Delta (pp)</th><th>N</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table>' if rows else '<p>Held-out evaluation has not yet produced results.</p>'}<p>Success requires fixture off, sustained lift, multiple contacting digits, and bounded drift. Simulation only; no self-collision or hardware calibration.</p></section></main></body></html>'''
(p/'report.html').write_text(doc)
print(p/'report.html')
