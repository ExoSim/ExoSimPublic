import numpy   as np
import quantities as pq
import os, time
from   ..classes import sed
from   ..lib     import exolib
from exosim.lib.exolib import exosim_msg

class zodiacal_light(object):
  def __init__(self, wl, level=1.0):
    exosim_msg('Instantiate Zodi ... ')
    st = time.time()
    spectrum = level*(3.5e-14*exolib.planck(wl, 5500*pq.K) + 
                      exolib.planck(wl, 270*pq.K) * 3.58e-8)
    
    self.sed 		= sed.Sed(wl, spectrum )
    self.transmission 	= sed.Sed(wl, np.ones(wl.size))
    self.units         = 'W m**-2 sr**-1 micron**-1'
    exosim_msg(' - execution time: {:.0f} msec.\n'.format((time.time()-st)*1000.0))