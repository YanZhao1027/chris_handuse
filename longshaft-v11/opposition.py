"""True loaded opposition, measured from rod contact normals, not visual poses."""
import numpy as np,mujoco as mj
def metrics(h):
    d=h.d;m=h.m;axis=d.xmat[h.oid].reshape(3,3)[:,0]
    nvec=np.zeros((2,3));forces=np.zeros(2);digits=set()
    selfpen=0.;objpen=0.;buf=np.zeros(6)
    for j in range(d.ncon):
        c=d.contact[j];a,b=int(c.geom1),int(c.geom2)
        if h.floor in (a,b):continue
        if h.ball not in (a,b):
            selfpen=max(selfpen,-float(c.dist));continue
        objpen=max(objpen,-float(c.dist))
        other=b if a==h.ball else a
        digit=h.geom_digit.get(other)
        if digit is None:continue
        mj.mj_contactForce(m,d,j,buf);f=max(0.,float(buf[0]))
        if f<.005:continue
        n=np.asarray(c.frame[:3])*(1. if a==h.ball else -1.)
        # Remove end-cap normals parallel to shaft; require genuine side contacts.
        n=n-axis*np.dot(n,axis)
        group=int(digit==4);nvec[group]+=f*n;forces[group]+=f*np.linalg.norm(n)
        if digit<4:digits.add(digit)
    lengths=np.linalg.norm(nvec,axis=1)
    cos=float(np.dot(nvec[0],nvec[1])/max(1e-9,lengths[0]*lengths[1])) if min(lengths)>.02 else 1.
    opposed=bool(cos<-.35 and min(forces)>.10 and len(digits)>=2)
    return dict(opposed=opposed,cos=cos,thumbForce=float(forces[1]),fingerForce=float(forces[0]),fingerDigits=len(digits),selfPen=max(0.,selfpen),objectPen=max(0.,objpen))
