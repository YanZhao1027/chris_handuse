"""Build paired replay data from actual held-out trajectories, never synthesize frames."""
import json
from pathlib import Path
run=Path('runs/ppo01')
data={}
for name in ['fixed','learned']:
    d=json.loads((run/(name+'_trajectory.json')).read_text())
    s=d['summary']; data[name]=dict(summary=dict(**s,**s['params'],duration=6.,releaseAt=None),frames=d['frames'])
    # The simulator terminates early on a drop; viewer holds the last true frame.
Path('tendonhand-learned/recordings.json').write_text(json.dumps(data,separators=(',',':')))
