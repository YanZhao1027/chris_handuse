import json
from pathlib import Path
for name in ['bc01','ppo01']:
 p=Path('runs')/name;h=p/'history.json'
 if h.exists():
  d=json.loads(h.read_text());print(name,'last',d[-1])
  for x in d:
   s=x.get('onlineValidation',x.get('validation'))
   if s:print(' validation',x.get('epoch',x.get('update')), 'stable20',s['stable20'],'final',s['finalSuccess'])
 b=p/'blind.json'
 if b.exists():
  d=json.loads(b.read_text());print(' blind', {k:v['summary'] for k,v in d['results'].items()})
