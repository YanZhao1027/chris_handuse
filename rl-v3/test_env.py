import unittest
import numpy as np
import mujoco as mj
from env import Hand,CFG,ACT,OBS,domain
class Tests(unittest.TestCase):
    def test_reference(self):
        h=Hand();o=h.reset(42,0,horizon=150)
        self.assertEqual(o.shape,(OBS,))
        for i in range(150):
            o,r,done,info=h.step(np.zeros(ACT))
            if i==24:self.assertTrue(info['fixture'])
            if i==25:self.assertFalse(info['fixture'])
            if done:break
        self.assertTrue(info['success'],info)
        self.assertGreater(info['finalHeight'],.26)
    def test_randomization(self):
        h=Hand()
        for seed in range(20):
            p=domain(seed);h.reset(seed)
            self.assertAlmostEqual(h.m.body_mass[h.oid],p['mass'])
            self.assertAlmostEqual(h.m.geom_size[h.ball,0],p['radius'])
            self.assertAlmostEqual(h.m.geom_friction[h.ball,0],p['friction'])
            self.assertTrue(np.allclose(h.relative(),p['center'],atol=1e-6))
            self.assertTrue(h.d.eq_active[h.eq])
    def test_control_effect(self):
        a,b=Hand(),Hand();a.reset(123,.3);b.reset(123,.3)
        for i in range(15):
            a.step(np.zeros(ACT));b.step(np.ones(ACT)*.5)
        self.assertFalse(np.allclose(a.d.ctrl,b.d.ctrl))
        self.assertGreater(np.linalg.norm(a.d.qpos-b.d.qpos),1e-3)
        self.assertTrue(np.isfinite(b.obs()).all())
    def test_repeatable(self):
        h=Hand();results=[]
        for j in range(2):
            h.reset(41,.8)
            for i in range(100):
                o,r,d,info=h.step(np.zeros(ACT))
                if d:break
            results.append(info)
        self.assertEqual(results[0],results[1])
if __name__=='__main__':unittest.main()
