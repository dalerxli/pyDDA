from __future__ import print_function
from numpy import arange, exp, ones, tensordot, eye, sqrt, zeros, complex128, array, pi, loadtxt
from pydda import dda

class dda_faster(dda):
  def __init__(self, i2xyz):
    dda.__init__(self, i2xyz)
    self.jk2d, self.jk2tdo = self.precomput(i2xyz)

  def precomput(self, i2xyz):
    """
      Precomputing some matrices for the interaction tensor
      self.i2xyz : array of coordinates of dipoles
    """
    n = len(i2xyz)
    self.jk2d   = zeros((n,n))
    self.jk2tdo = zeros((n,n,3,3))
    for j,rj in enumerate(i2xyz):
      for k,rk in enumerate(i2xyz):
        if j!=k:
          vjk = rj - rk
          d   = sqrt((vjk*vjk).sum())
          self.jk2d[j,k] = d
          rjk = vjk/d
          tdo = tensordot(rjk, rjk, axes=0)
          self.jk2tdo[j,k,:,:] = tdo
    return self.jk2d,self.jk2tdo

  def interaction_tensor_precomput(self, i2alpha, ks=0.0):
    """
      The interaction tensor for a given frequency from https://doi.org/10.1364/JOSAA.11.001491
      ks = omega / c
      i2alpha : array of dipole's polarizabilities
      self.i2xyz : array of coordinates of dipoles
    """
    i2xyz = self.i2xyz
    assert self.n == i2alpha.shape[0]
    n = self.n
    Ajk = zeros((n,3,n,3), dtype=complex128)
    i3  = eye(3)
    k2 = ks**2
    for j,rj in enumerate(i2xyz):
      for k,rk in enumerate(i2xyz):
        if j==k:
          Ajk[j,:,k,:] = 1.0/i2alpha[j]
        else:
          vjk = rj - rk
          djk = self.jk2d[j,k]
          tdo = self.jk2tdo[j,k]
          Ajk[j,:,k,:] = exp(1j*ks*djk)/djk*( k2*(tdo-i3) + (1j*ks*djk-1.0)/djk**2 * (3*tdo-i3) )
    return Ajk

  interaction_tensor = interaction_tensor_precomput


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import cProfile
  ww = arange(0.01, 7.0, 0.02)
  coords = 0.8*loadtxt('ag54.xyz', usecols=(1,2,3), skiprows=2)*0.529177210564
  print(coords)
  c1 = dda(coords)
  #cProfile.run('c1.absorption_dipole(ww, d=1.0, wp=6.0, epsilon_infinity=4.0)')
  abs_dip1 = c1.absorption_dipole(ww, d=1.0, wp=6.0, epsilon_infinity=4.0)

  #plt.plot(ww, abs_dip1.imag, label='1')
  #plt.legend()
  #plt.show()
