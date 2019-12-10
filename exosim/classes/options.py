import xml.etree.ElementTree as ET
import numpy as np
import quantities as aq
import os
import sys, os
from exosim.lib.exolib import exosim_msg
from exosim.lib.exolib import exosim_error

from   ..  import __path__
XML_EXOSIM_DEFAULTS        = 'exosim_defaults.xml'

class Entry(object):
  val       = None
  attrib    = None
  xml_entry = None
  
  def __call__(self):
    return self.val
    
  def parse(self, xml):
    self.attrib = xml.attrib
    for attr in self.attrib.keys():
      setattr(self, attr, self.attrib[attr])
      
    if hasattr(self, 'units'):
      try:
        self.val = aq.Quantity(map(float, self.val.split(',')), self.units).simplified
        if self.units == 'deg': self.val = [x*aq.rad for x in self.val] # workaround for qt unit coversion
        if len(self.val) == 1: self.val = self.val[0]
      except (ValueError, LookupError):
        print 'unable to convert units in entry [tag, units, value]: ', \
                                 xml.tag, self.units, self.val

class Options(object):
  opt = None
  MSG_PREFIX = 'Options.class'
  def __init__(self, filename = None, default_path = None):
    
    if not(filename): filename = XML_EXOSIM_DEFAULTS
    
    self.opt = self.parser(ET.parse(filename).getroot())
    
    if default_path:
      setattr(self.opt, "__path__", default_path)
    elif hasattr(self.opt.common, "ConfigPath"):
      setattr(self.opt, "__path__", 
            os.path.expanduser(self.opt.common.ConfigPath().replace('__path__', __path__[0])))
    else:
      exosim_error("Path to config files not defined")
    
    
    self.validate_options()
    self.calc_metaoptions()
    
  def parser(self, root):
    obj = Entry()
    
    for ch in root: 
      retval = self.parser(ch)
      retval.parse(ch)
      
      if hasattr(obj, ch.tag):
        if isinstance(getattr(obj, ch.tag), list):
          getattr(obj, ch.tag).append(retval)
        else:
          setattr(obj, ch.tag,  [getattr(obj, ch.tag), retval])
      else:
        #retval.val = retval.val.replace('__path__', __path__[0]) if isinstance(retval(), str) else retval.val
        setattr(obj, ch.tag, retval)
    return obj
  
  
  def validate_options(self):
    self.validate_is_list() 
    self.validate_qe_rms_matrix_file() ## Depends on validate_is_list() 
    self.validate_True_False_spelling()
      
  def validate_qe_rms_matrix_file(self):
    for chan in self.opt.channel:
        pathToQeArr = chan.qe_rms_matrix_file()
        if pathToQeArr.strip().lower() != 'none':
          if not(os.path.isfile(pathToQeArr)):
            raise IOError("'qe_rms_matrix_file' file does not exist")
          else:
            temp = np.loadtxt(pathToQeArr, delimiter = ',')
            if temp.shape[0] != int(chan.array_geometry()[0]) and \
                    temp.shape[1]!= int(chan.array_geometry()[1]):
              raise ValueError('matrix dimensions in "qe_rms_matrix_file" file do not match "array_geometry"')
  def validate_is_list(self):
    if not isinstance(self.opt.common_optics.optical_surface, list):
      self.opt.common_optics.optical_surface = [self.opt.common_optics.optical_surface]
    if not isinstance(self.opt.channel, list):
      self.opt.channel = [self.opt.channel]
    
  def validate_True_False_spelling(self):
    acceptedValues = ['True', 'False']
    testCases = [
                 'astroscene/transit_is_primary',
                 'noise/EnableSpatialJitter',
                 'noise/EnableSpectralJitter',
                 'noise/EnableShotNoise',
                 'noise/EnableReadoutNoise',
                 ]
    for item in testCases:
      if not self.opt.__getattribute__(item.split('/')[0]).__dict__[item.split('/')[1]]() in acceptedValues:
        raise ValueError("Accepted values for [%s] are 'True' or 'False'"%item)
  
  
  def calc_metaoptions(self):
    self.calc_metaoption_wl_delta()
    self.calc_metaoption_qe_rms_matrix()
    
  def calc_metaoption_wl_delta(self):
     wl_delta = self.opt.common.wl_min()/self.opt.common.logbinres()
     setattr(self.opt.common, 'common_wl', (np.arange(self.opt.common.wl_min(),
                   self.opt.common.wl_max(),
                   wl_delta)*wl_delta.units).rescale(aq.um))
     
  def calc_metaoption_qe_rms_matrix(self):
    for chan in self.opt.channel:
      pathToQeArr = chan.qe_rms_matrix_file()
      if pathToQeArr.lower() != 'none':
        qe_rms_matrix = np.loadtxt(pathToQeArr, delimiter = ',')
      else:
        qe_rms_matrix = np.ones(  (  np.int(chan.array_geometry()[0]), 
                                     np.int(chan.array_geometry()[1])  )  )
      setattr(chan, 'qe_rms_matrix', qe_rms_matrix * aq.dimensionless)
  


if __name__ == "__main__":
  opt = Options()