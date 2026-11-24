import time,numpy as np
from env import Hand,ACT
h=Hand();t=time.perf_counter()
for difficulty in [0,.3,1.]:
    rr=[]
    for seed in range(16):
        o=h.reset(seed,difficulty)
        for k in range(h.horizon):
            o,r,d,i=h.step(np.zeros(ACT))
            if d:break
        rr.append(i['success'])
    print('baseline',difficulty,sum(rr),'/',len(rr),flush=True)
print('elapsed',time.perf_counter()-t)
