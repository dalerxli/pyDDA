from __future__ import print_function
from numpy import arange, exp, ones, tensordot, eye, sqrt, zeros, complex128, array, pi, loadtxt, einsum
from pydda import dda

class dda_faster(dda):
  def __init__(self, i2xyz):
    dda.__init__(self, i2xyz)
    self.precomput(i2xyz)

  def precomput(self, i2xyz):
    """
      Precomputing some matrices for the interaction tensor
      self.i2xyz : array of coordinates of dipoles
    """
    n = len(i2xyz)
    self.jk2d   = zeros((n,n,3,3))
    self.jk2tdo = zeros((n,n,3,3))

    self.jk2d1  = zeros((n,3,n,3))
    self.jk2tdo3_m_i3 = zeros((n,3,n,3))
    self.jk2tdo1_m_i3 = zeros((n,3,n,3))
    i3  = eye(3)
    for j,rj in enumerate(i2xyz):
      for k,rk in enumerate(i2xyz):
        if j!=k:
          vjk = rj - rk
          d   = sqrt((vjk*vjk).sum())
          rjk = vjk/d
          tdo = tensordot(rjk, rjk, axes=0)
          self.jk2d[j,k] = d
          self.jk2tdo[j,k] = tdo
          self.jk2d1[j,:,k,:] = d
          self.jk2tdo3_m_i3[j,:,k,:] = 3.0*tdo-i3
          self.jk2tdo1_m_i3[j,:,k,:] = 1.0*tdo-i3
        else:
          self.jk2d[j,k] = 1.0
          self.jk2tdo[j,k] = i3
          self.jk2d1[j,:,k,:] = 1.0
          self.jk2tdo3_m_i3[j,:,k,:] = -i3
          self.jk2tdo1_m_i3[j,:,k,:] = -i3

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
    Ajk = zeros((n,n,3,3), dtype=complex128)
    i3  = eye(3)
    k2 = ks**2
    jk2d = self.jk2d
    jk2tdo = self.jk2tdo
    ttdo_m_i3 = 3.0*jk2tdo - i3
    otdo_m_i3 = jk2tdo - i3
    Ajk = exp(1j*ks*jk2d)/jk2d*(k2*otdo_m_i3+(1j*ks*jk2d-1.0)/jk2d**2*ttdo_m_i3)
    for i in range(n): Ajk[i,i,:,:] = 1.0/i2alpha[i]*i3
    return einsum('jkab->jakb', Ajk)

  interaction_tensor = interaction_tensor_precomput


if __name__ == '__main__':
  import matplotlib.pyplot as plt
  import cProfile
  ww = arange(0.01, 7.0, 0.02)
  coords = 0.8*loadtxt('ag147.xyz', usecols=(1,2,3), skiprows=2)*0.529177210564
  print(coords)
  c1 = dda_faster(coords)
  #cProfile.run('c1.absorption_dipole(ww, d=1.0, wp=6.0, epsilon_infinity=4.0)')
  abs_dip1 = c1.absorption_dipole(ww, d=1.0, wp=6.0, epsilon_infinity=4.0)

  plt.plot(ww, abs_dip1.imag, label='1')
  plt.legend()
  plt.show()
