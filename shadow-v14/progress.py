import json
from pathlib import Path
p=Path('runs/ppo01');h=p/'history.json'
if not h.exists():print('Initializing GPU environment and CPU validation bank; no completed update yet.')
else:
 d=json.loads(h.read_text());r=d[-1]
 print('update',r['update'],'/900; interactions',r['steps'],'physics',r['physicsSteps'],'steps/s',round(r['stepsPerSecond']),'goal range',r['goalMin'],r['goalMax'])
 for q in reversed(d):
  if q.get('episodes'):print('latest completed episodes',q['episodes'],'success',q['success'],'retained',q['retained'],'negative',q['negative'],'positive',q['positive'],'frontier',q.get('frontierSuccess'),'second-plane',q.get('multiSuccess'),'successful recontacts',q.get('successfulRecontactEpisodes'));break
 for q in reversed(d):
  if 'validation' in q:print('CPU validation update',q['update'],q['validation']);break
 if (p/'blind.json').exists():
  for t in json.loads((p/'blind.json').read_text())['tests']:print('blind',t['seed'],{k:v['summary'] for k,v in t['results'].items()})
