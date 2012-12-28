# X-ray tools
# Sam Tardif
# samuel.tardif@gmail.com

from numpy import *
from numpy.linalg.linalg import *
import spec_reader as sr
import matplotlib 

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

 
def uncertainty_fr(p,n,po,no):
  """
  Returns the flipping ratio (fr) quantity, defined as 
    (+)-(-)/(+)+(-)
  as well as the Poisson-based uncertainty (dfr)
  
  p  = detector intensity for (+) orientation
  n  = detector intensity for (-) orientation
  po = monitor intensity for (+) orientation
  no = monitor intensity for (-) orientation
  """
  
  dp, dn, dpo, dno = sqrt(p), sqrt(n), sqrt(po), sqrt(no)
  #dp, dn, dpo, dno = sqrt(p), sqrt(n), po.std(), no.std() # Non-Poisson statistics
  
  dfr = no/po/n*2./(no/po*p/n+1.)**2*dp +\
        n/p/no*2./(po/no*n/p+1.)**2*dpo +\
  	   	po/no/p*2./(po/no*n/p+1.)**2*dn +\
	    	p/po/n*2./(no/po*p/n+1)**2*dno
  
  fr = (p/po-n/no)/(p/po+n/no)
  
  return dfr, fr   


def xprplot_th(spec_file, scan_number, detector, monitor, centered=True, do_plot=True, save=False, auto_close=False):
  """
  Process the XPR theta scans by determining the center of the XPR Bragg peak and calculating the flipping ratio symmetrically.
  Optional: plot the results
  
  'centered'   : True if the scan was measured symmetrically to the XPR Bragg peak center  # False case not implemented yet
  'do_plot'    : True if the results are to be plotted
  'save'       : True if the plots are to be saved (unused if 'do_plot' = False). Default name is "<spec_file>_scan<scan_number>_<detector>"  (.png, .pdf and .Semf)
  'auto_close' : True if the plot windows should be closed automatically, useful for batch processing (unused if 'do_plot' = False)
  """
  
  s = sr.Scan(spec_file,scan_number)
  xprth = getattr(s,s.counters[0]) # depending on the xpr used (1 or 2), x axis is xpr1th or xpr2th
  
  if centered:
    midpoint = xprth.shape[0]/2
    if xprth.shape[0]%2 == 1 : # odd number of points in the scan
      x  = xprth[midpoint+1:]-xprth[midpoint]
      n  = getattr(s,detector)[:midpoint][::-1]
      p  = getattr(s,detector)[midpoint+1:]
      no = getattr(s,monitor)[:midpoint][::-1]
      po = getattr(s,monitor)[midpoint+1:]
      dfr, fr = uncertainty_fr(p,n,po,no)
      #print 'midpoint is %f'%(midpoint)
    
    else: # even number of points in the scan (center point not measured)
      x  = xprth[midpoint:]-(xprth[0]+xprth[-1])/2.
      n  = getattr(s,detector)[:midpoint][::-1]
      p  = getattr(s,detector)[midpoint:]
      no = getattr(s,monitor)[:midpoint][::-1]
      po = getattr(s,monitor)[midpoint:]      
      dfr, fr = uncertainty_fr(p,n,po,no)
  
  
  if do_plot:
    initial_font_size = matplotlib.rcParams.get('font.size')
    matplotlib.rcParams.update({'font.size': 16})
    matplotlib.pyplot.figure()
    matplotlib.pyplot.errorbar(x,fr,yerr=dfr,fmt='bo-')
    matplotlib.pyplot.xlabel('xpr offset (degree)')
    matplotlib.pyplot.ylabel('Flipping ratio (detector = %s, monitor = %s)'%(detector, monitor))
    matplotlib.pyplot.title(spec_file + ' scan #%i'%(scan_number))
    matplotlib.pyplot.hlines(0,x.min(),x.max())
    matplotlib.pyplot.grid()
    matplotlib.pyplot.show()
    if save:
      matplotlib.pyplot.savefig(spec_file + '_scan%i_%s.pdf'%(scan_number,detector))
      matplotlib.pyplot.savefig(spec_file + '_scan%i_%s.emf'%(scan_number,detector))
      matplotlib.pyplot.savefig(spec_file + '_scan%i_%s.png'%(scan_number,detector),dpi=150)
    if auto_close:
      matplotlib.pyplot.close()
    matplotlib.rcParams.update({'font.size': initial_font_size})
    
    # if do_plot:
    # figure()
    # errobar(x,fr,yerr=dfr,fmt='bo-')
    # xlabel('xpr offset (degree)')
    # ylabel('Flipping ratio (detector = %s, monitor = %s)'%(detector, monitor))
    # title(spec_file + ' scan #%i'%(scan_number))
    # hlines(0,x.min(),x.max())
    # grid()
    # show()
    # if save:
      # savefig(spec_file + '_scan%i_%s.pdf'%(scan_number,detector))
      # savefig(spec_file + '_scan%i_%s.emf'%(scan_number,detector))
      # savefig(spec_file + '_scan%i_%s.png'%(scan_number,detector),dpi=150)
    # if auto_close:
      # close()
    
  return x,fr,dfr
  
  
class XprScan(sr.Scan):
  """
  XprScan class to include the flipping ratio and the theoretical error.
  Attributes are inherited from the `spec_reader.Scan` class.
  Flipping ratio and error can calculated and added for any counter using the `XprScan.fr` method.
  """
 
  def fr(self, detector, monitor, centered=True, do_plot=False, save=False, auto_close=False):
    """
    Method to calculate the flipping ratio and estimated error for any counter (detector), given a monitor.
    Calculated flipping ratio and error are added as new attributes `XprScan.<detector>_fr` and `XprScan.<detector>_dfr` respectively.
    The xpr offset axis is also added as `XprScan.xpr_offset`
    """
    xpr_offset, fr,  dfr  = xprplot_th(self.file, self.number, detector, monitor, centered=centered, do_plot=do_plot, save=save, auto_close=auto_close)
    setattr(self, 'xpr_offset', xpr_offset)
    setattr(self, detector+'_fr', fr)
    setattr(self, detector+'_dfr', dfr)