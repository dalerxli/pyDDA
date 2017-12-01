from __future__ import print_function
from numpy import arange, exp, ones, tensordot, eye, sqrt, zeros, complex128, array, pi

class dda():
  def __init__(self, i2xyz):
    assert i2xyz.shape[1] == 3
    self.n = i2xyz.shape[0]
    self.i2xyz = i2xyz
    self.c = 137.0
    
  def interaction_tensor_loops(self, i2alpha, k=0.0):
    """
      The interaction tensor for a given frequency from https://doi.org/10.1364/JOSAA.11.001491
      k = omega / c
      i2alpha : array of dipole's polarizabilities
      i2xyz : array of coordinates of dipoles
    """
    i2xyz = self.i2xyz
    assert self.n == i2alpha.shape[0]
    n = self.n
    Ajk = zeros((n,3,n,3), dtype=complex128)
    i3  = eye(3)
    k2 = k**2
    for j,rj in enumerate(i2xyz):
      for k,rk in enumerate(i2xyz):
        if j==k:
          Ajk[j,:,k,:] = 1.0/i2alpha[j]
        else:
          vjk = rj - rk
          djk = sqrt((vjk*vjk).sum())
          rjk = vjk/djk
          tdo = tensordot(rjk, rjk, axes=0)
          Ajk[j,:,k,:] = exp(1j*k*djk)/djk*( k2*(tdo-i3) + (1j*k*djk-1.0)/djk**2 * (3*tdo-i3) )
    
    return Ajk
  
  interaction_tensor = interaction_tensor_loops
    
  def polarization_direct_solver(self, i2alph, k=0.0, i2einc=None):
    from numpy.linalg import solve 
    """ 
      Compute the polarization P (on each dipole) for a given polarizability alpha (for each dipole) and
      given field strength of incident electric field Einc (for each dipole)
    """
    n = self.n
    i2einc = array([1.0, 0.0, 0.0]*n).reshape((n,3)) if i2einc is None else i2einc    

    assert 3 == i2einc.shape[1]
    assert n == i2einc.shape[0]
    assert n == i2alph.shape[0]

    Ajk = self.interaction_tensor(i2alph, k).reshape((n*3,n*3))
    i2pol = solve(Ajk, i2einc.reshape(n*3))
    return i2pol.reshape((n,3))

  polarization = polarization_direct_solver

  def epsilon_drude(self, ww, wp=1.0, eta=None, epsilon_infinity=1.0, **kw):
    """ Compute a Drude dielectruc function eps = eps(-infinity) - wp^2/(-w^2+i*eta*w) """
    eta = ww[1]-ww[0] if eta is None else eta
    return epsilon_infinity - wp**2 / (ww**2+1j*eta)

  def alpha_clausius_mossotti(self, iw2epsilon, d=1.0, **kw):
    """ Polarizability of a sphere from the dielectric function of the material, according to Clausius-Mossotti """
    return 3*d**3/(4*pi) * (iw2epsilon - 1.0)/(iw2epsilon + 2.0)
  
  def absorption_dipole(self, ww, **kw):
    self.ww  = ww
    self.eps = self.epsilon_drude(ww, **kw)
    self.alp = self.alpha_clausius_mossotti(self.eps, **kw)
    self.abs_dip = zeros(len(self.ww), dtype=complex128)
    i2alph = zeros(self.n, dtype=complex128)
    for iw,(w,alp) in enumerate(zip(ww,self.alp)):
      i2alph[:] = alp
      i2pol = self.polarization(i2alph)
      self.abs_dip[iw] = (i2pol*alp*i2pol.conj()).sum()
    return self.abs_dip
    
if __name__ == '__main__':
  import matplotlib.pyplot as plt
  ww = arange(0.01, 1.0, 0.005)

  coords = array([[1.0, 0.01, 0.0], [-1.0, 0.0, 0.0],[1.0, 0.0, 0.0]])
  c2 = dda(coords)
  abs_dip2 = c2.absorption_dipole(ww, d=5.0)

  #plt.plot(ww, abs_dip2.real, label='abs1')
  plt.plot(ww, abs_dip2.imag, label='abs2')
  plt.legend()
  plt.show()
