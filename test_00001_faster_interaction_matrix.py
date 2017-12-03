from __future__ import print_function, division
import os,unittest,numpy as np


class KnowValues(unittest.TestCase):

  def test_fast_inter_mat(self):
    """ Test precomputed interaction """
    from pydda import dda as dda
    from pydda_faster import dda_faster
    from timeit import default_timer as timer

    ww = np.arange(0.01, 7.0, 0.2)
    fname = 'ag147.xyz'
    coords = np.loadtxt(fname, usecols=(1,2,3), skiprows=2)*0.529177210564
    #print(coords)
    print(fname)
    c1 = dda(coords)
    c2 = dda_faster(coords)
    i2alpha = np.ones(c1.n, dtype=np.complex128)
    t1 = timer()
    ajk1 = c1.interaction_tensor(i2alpha, ks=4.0)
    t2 = timer(); print(' interaction time with simple loops ', t2-t1); t1=t2
    
    ajk2 = c2.interaction_tensor(i2alpha, ks=4.0)
    #print('abs(ajk1-ajk2).sum()/ajk1.size', abs(ajk1-ajk2).sum()/ajk1.size)
    t2 = timer(); print(' interaction time with precomput mat', t2-t1); t1=t2
    
    self.assertTrue(np.allclose(ajk1, ajk2))

if __name__ == "__main__": unittest.main()
