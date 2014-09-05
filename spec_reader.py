# SPEC reader
# 2014-09-05
# Sam Tardif (samuel.tardif@gmail.com)

import numpy as np

class Scan:
  """ Simple class to read SPEC files and extract scans. All the parameters of the scan and the data are read and stored as attributes.
  
  Definition:
  -----------
  Scan(spec_file, scan_number, verbose = False)
   > spec_file as string
   > scan_number as integer
  
  Attributes:
  -----------
  <countername>...data in counter <countername> (see counters for description)
  number..........scan number
  type............scan type
  args............scan arguments (motor start stop npoints counting)
  date............scan start datestamp
  ct..............scan counting time per point
  Qo..............H K L position at start of scan
  N...............number of counters
  counters........list of counters
  motors..........dictionary of motors and their initial position
  comments........all comments
  
  Examples:
  --------
  # read the scan
  In : scan = Scan('./lineup0.dat', 265)
  
  # learn about the motors
  In : scan.motors
  Out:  
  {'Chi': -144.8913,
   'Phi': 176.9778,
   ...
   'xpr3z': -8.8}
  
  # extract the info on a particular motor
  In : scan.motors['tth']
  Out: 35.042
  
  # plot two counters vs each others
  In : plot(scan.th,scan.det/scan.IC1)
  
  
  
  """
  
  verbose = False

  # routine to search for a particular string in the file
  def __findstring__(self, file, pattern):
    found = False
    l = file.readline()
    if l != '' :
      while not found:
        l = file.readline()
        if l == '' : break
        elif len(l) >= len(pattern):
          if l[:len(pattern)] == pattern: found = True
    if l == '': raise NameError
    else : return l

  
  # dictionary definitions for handling the spec identifiers
  def __param__(self):
    return  { 'S' : self.__scanline__,
              'D' : self.__dating__,
              'T' : self.__counting__,
              'G' : self.__configurating__,
              'Q' : self.__hkl__,
              'O' : self.__motorslabeling__,
              'P' : self.__positioning__,
              'N' : self.__speccol__,
              'L' : self.__counterslabeling__,
              'C' : self.__commenting__}
  
  def __scanline__(self, l):
    self.number = int(l.split()[1])
    self.type = l.split()[2]
    self.args = l.split()[3:]  

  def __dating__(self, l):
    self.date = l[3:]

  def __counting__(self, l):
    self.ct = float(l.split()[1])
    self.ct_units = l.split()[2]

  def __configurating__(self, l):
    self.__config__ = self.__config__ + l[3:] + " "

  def __hkl__(self, l):
    self.Qo = l[3:].split()

  def __motorslabeling__(self, l):
    self.__motorslabels__ = self.__motorslabels__ + l[3:] + " "

  def __positioning__(self, l):
    self.__positions__ = self.__positions__ + l[3:] + " "
  
  def __speccol__(self, l):
    self.N = int(l[3:])
  
  def __counterslabeling__(self, l):
    self.counters = l.split()[1:]
  
  def __commenting__(self, l):
    self.comments = self.comments + l[3:]
  

  
  
  
  def __init__(self, spec_file, scan_number, verbose = verbose):
    self.__motorslabels__ = "" # list of all motors in the experiement
    self.__positions__ = "" # list the values of all motors
    self.__config__ = "" # list the values of the UB matrix config
    self.comments = ""
    self.file = spec_file
    
    f = open(spec_file, 'r') 

    #first read the file header (mostly comments and motors definiton)
    l = f.readline()
    if verbose : print l
    while l[0] == '#':
      try: self.__param__()[l[1]](l)
      except KeyError:
        if verbose : print "unprocessed line:\n" + l
      l = f.readline()


    #now try to find the scan 
    if verbose  : print "searching scan " + l
    try: 
      l = self.__findstring__(f,"#S %i"%(scan_number))
      
      #read the scan header 
      while l[0] == '#':
        try: self.__param__()[l[1]](l)
        except KeyError:
          if verbose : print "unprocessed line:\n" + l
        l = f.readline()
      
      
      #finally read the data (comments at the end are also read and added to the comment attribute)
      data = [map(float,l.split())]
      l = f.readline()
      while l != '\n' and l != '':
        if l[0] == '#' and l[1] != 'C':
          break
  #      print l + 'lenght = %i'%(len(l))
        if l[0] == '#':
          try: self.__param__()[l[1]](l)
          except KeyError:
            if verbose : print "unprocessed line:\n" + l
        else : data.append(map(float,l.split()))
        l = f.readline()
      
      
      #set the data as attributes with the counter name
  #    print data
      for i in xrange(len(self.counters)):
        setattr(self, self.counters[i], np.asarray(data)[:,i])
      
      
      #make the motors/positions dictionary
      self.motors = dict(zip(self.__motorslabels__.split(), map(float,self.__positions__.split())))
      
      
      #TEST : attibute-like dictionary
      # removed due to conflicts when a motor was also a counter
  #    for motor in self.motors:
  #                setattr(self, motor, self.motors[motor])
      
      
      #small sanity check, sometimes N is diffrent from the actual number of columns
      #which is known to trouble GUIs like Newplot and PyMCA
      if self.N != len(self.counters): 
        print "Watch out! There are %i counters in the scan but SPEC knows only N = %i !!"%(len(self.counters),self.N)
      
    except NameError:
      print "Scan number not found!"
      
    f.close()  
