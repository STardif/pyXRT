# X-ray tools
# Sam Tardif
# samuel.tardif@gmail.com

from numpy import *
from numpy.linalg.linalg import *

def E2L(energy):
  """
  Calculates the wavelength (in m) corresponding to the energy (in eV)
  using h = 4.135667516E-15 eV.s
  and   c = 2.99792458E8 m/s
  
  Numerical values from P.J. Mohr, B.N. Taylor, and D.B. Newell (2011), "The 2010 CODATA Recommended Values of the Fundamental Physical Constants" (Web Version 6.0). This database was developed by J. Baker, M. Douma, and S. Kotochigova. Available: http://physics.nist.gov/constants [Thursday, 02-Jun-2011 21:00:12 EDT]. National Institute of Standards and Technology, Gaithersburg, MD 20899.
  """
  return 4.135667516E-15*2.99792458E8/energy
  
  
  
  
def L2E(wavelength):
  """
  Calculates the energy (in eV) corresponding to the wavelength (in m)
  using h = 4.135667516E-15 eV.s
  and   c = 2.99792458E8 m/s
  
  Numerical values from P.J. Mohr, B.N. Taylor, and D.B. Newell (2011), "The 2010 CODATA Recommended Values of the Fundamental Physical Constants" (Web Version 6.0). This database was developed by J. Baker, M. Douma, and S. Kotochigova. Available: http://physics.nist.gov/constants [Thursday, 02-Jun-2011 21:00:12 EDT]. National Institute of Standards and Technology, Gaithersburg, MD 20899.
  """
  return 4.135667516E-15*2.99792458E8/wavelength
  
  
  
  
def tth(H,E,a):
  """
  Calculates the 2theta angle (degree) for the reflection vector H (r.l.u.) at energy E (eV) for a cubic crystal of lattice parameter a (m) 
  Example:
  >>> tth((0,0,10),10871,10.015E-10)
  >>> 69.416502777784046
  """
  return 2*arcsin(E2L(E)/(2*a/norm(H)))*180./pi