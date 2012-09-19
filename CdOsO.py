# calculate the x-ray scattering in Cd2Os2O7 
# S. Tardif (samuel.tardif@gmail.com)

# NOTE : this script requires the atomic scattering factors (from www.cxro.org)
atomic_scattering_factors_folder = './atomic_scattering_factors/'

E=8000. # default energy (in eV)

x=0.319 # x parameter for O1

base = array([\
[ 0.00, 0.00, 0.00],
[ 0.00, 0.50, 0.50],
[ 0.50, 0.00, 0.50],
[ 0.50, 0.50, 0.00]])

#16d
Cd = array([\
[ 5./8, 5./8, 5./8],
[ 3./8, 7./8, 1./8],
[ 7./8, 1./8, 3./8],
[ 1./8, 3./8, 7./8]])

#16c
Os = array([\
[ 1./8, 1./8, 1./8],
[ 7./8, 3./8, 5./8],
[ 3./8, 5./8, 7./8],
[ 5./8, 7./8, 3./8]])

#48f
O1 = array([\
[      x,       0,       0],	
[     -x,    1./2,    1./2],	
[     0.,       x,       0],	
[   1./2,      -x,    1./2],	
[      0,       0,       x],	
[   1./2,    1./2,      -x],
[   3./4,  x+1./4,    3./4],	
[   1./4, -x+1./4,    1./4],	
[ x+3./4,    1./4,    3./4],	
[-x+3./4,    3./4,    1./4],	
[   3./4,    1./4, -x+3./4],	
[   1./4,    3./4,  x+3./4]])

#8b
O2 = array([\
[ 1./2, 1./2, 1./2],
[ 1./4, 3./4, 1./4]])



def afs(se,E):
  """Returns the complex atomic scattering factor
  se = scattering element, E = Energy in eV
  
  Example : 
  --------
  afs("Cd",10520) 
  >>> (47.99187254901961+2.9819198039215684j)
  """
  f = loadtxt(atomic_scattering_factors_folder+se.lower()+'.nff',skiprows=1) # Energy, f1, f2
  return interp(E, f[:,0], f[:,1])+1.0j*interp(E, f[:,0], f[:,2])

  
allCd = Cd
for i in xrange(len(base)-1):
  allCd = append(allCd, Cd+(base[i+1]*ones((4,3))),axis=0)
#allCd = allCd%1

allOs = Os
for i in xrange(len(base)-1):
  allOs = append(allOs, Os+(base[i+1]*ones((4,3))),axis=0)

allO1 = O1
for i in xrange(len(base)-1):
  allO1 = append(allO1, O1+(base[i+1]*ones((12,3))),axis=0)

allO2 = O2
for i in xrange(len(base)-1):
  allO2 = append(allO2, O2+(base[i+1]*ones((2,3))),axis=0)

allO = append(allO1, allO2, axis=0)  
  
def F(h,k,l, E=E):
  H = array([h,k,l])
  return afs("Cd", E)*exp(-1.0j*2*pi*dot(allCd,H)).sum()+\
         afs("Os", E)*exp(-1.0j*2*pi*dot(allOs,H)).sum()+\
		 afs("O" , E)*exp(-1.0j*2*pi*dot(allO, H)).sum()
   
def F2(h,k,l, E=E):
  return abs(F(h,k,l,E))**2
   
print "111 %f"%(F2(1,1,1))
print "311 %f"%(F2(3,1,1))
print "222 %f"%(F2(2,2,2))
print "400 %f"%(F2(4,0,0))
print "331 %f"%(F2(3,3,1))
print "422 %f"%(F2(4,2,2))
print "511 %f"%(F2(5,1,1))
print "440 %f"%(F2(4,4,0))
print "531 %f"%(F2(5,3,1))
print "620 %f"%(F2(6,2,0))
print "533 %f"%(F2(5,3,3))
print "622 %f"%(F2(6,2,2))
print "444 %f"%(F2(4,4,4))
print "711 %f"%(F2(7,1,1))
print "731 %f"%(F2(7,3,1))