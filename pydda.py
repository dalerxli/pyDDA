from __future__ import print_function
from numpy import arange, exp, ones, tensordot, eye, sqrt, zeros, complex128, array, pi, loadtxt

class dda():
  def __init__(self, i2xyz):
    assert i2xyz.shape[1] == 3
    self.n = i2xyz.shape[0]
    self.i2xyz = i2xyz
    self.c = 137.0
    
  def interaction_tensor_loops(self, i2alpha, ks=0.0):
    """
      The interaction tensor for a given frequency from https://doi.org/10.1364/JOSAA.11.001491
      ks = omega / c
      i2alpha : array of dipole's polarizabilities
      i2xyz : array of coordinates of dipoles
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
          Ajk[j,:,k,:] = i3*1.0/i2alpha[j]
        else:
          vjk = rj - rk
          djk = sqrt((vjk*vjk).sum())
          rjk = vjk/djk
          tdo = tensordot(rjk, rjk, axes=0)
          Ajk[j,:,k,:] = exp(1j*ks*djk)/djk*( k2*(tdo-i3) + (1j*ks*djk-1.0)/djk**2 * (3*tdo-i3) )
    
    return Ajk
  
  interaction_tensor = interaction_tensor_loops

  def polarization_direct_solver(self, i2alph, i2einc, k=0.0):
    from numpy.linalg import solve 
    """ 
      Compute the polarization P (on each dipole) for a given polarizability alpha (for each dipole) and
      given field strength of incident electric field Einc (for each dipole)
    """
    n = self.n
    assert 3 == i2einc.shape[1]
    assert n == i2einc.shape[0]
    assert n == i2alph.shape[0]

    Ajk = self.interaction_tensor(i2alph, k).reshape((n*3,n*3))
    i2pol = solve(Ajk, i2einc.reshape(n*3))
    return i2pol.reshape((n,3))

  polarization = polarization_direct_solver

  def epsilon_drude(self, ww, wp=1.0, eta=None, epsilon_infinity=1.0, **kw):
    """ Compute a Drude dielectruc function eps = eps(-infinity) - wp^2/(w^2+i*eta*w) """
    eta = (ww[1]-ww[0])*2 if eta is None else eta
    return epsilon_infinity - wp**2 / (ww**2+1j*eta)

  def alpha_clausius_mossotti(self, iw2epsilon, d=1.0, **kw):
    """ Polarizability of a sphere from the dielectric function of the material, according to Clausius-Mossotti """
    return 3*d**3/(4*pi) * (iw2epsilon - 1.0)/(iw2epsilon + 2.0)
  
  def absorption_dipole(self, ww, **kw):
    self.ww  = ww
    self.w2eps = self.epsilon_drude(ww, **kw)
    self.w2alp = self.alpha_clausius_mossotti(self.w2eps, **kw)
    self.e_inc = array([1.0, 0.0, 0.0])
    self.abs_dip = zeros(len(self.ww), dtype=complex128)

    i2alph = zeros(self.n, dtype=complex128)
    i2einc = zeros((self.n,3), dtype=complex128)
    for iw,(w,alp) in enumerate(zip(ww,self.w2alp)):
      i2alph[:],i2einc[:] = alp, self.e_inc
      i2pol = self.polarization(i2alph, i2einc)
      self.abs_dip[iw] = (i2einc.conj()*i2pol).sum()
    return self.abs_dip
    
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
