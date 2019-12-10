import numpy           as     np
import pylab           as     pl
import sys, time, os
import exosim
from exosim.lib.exolib import exosim_msg

def run_exosim(opt=None):
  
  star, planet = exosim.modules.astroscene.run(opt)
  
  exosim_msg(' Stellar SED: {:s}\n'.format(os.path.basename(star.ph_filename)))
  exosim_msg(' Star luminosity {:s}\n'.format(star.luminosity))
  
  #Instanciate Zodi
  zodi = exosim.classes.zodiacal_light(opt.common.common_wl, level=1.0)
  
  exosim.exolib.sed_propagation(star.sed, zodi.transmission)
  #Run Instrument Model
  channel = exosim.modules.instrument.run(opt, star, planet, zodi)
  #Create Signal timelines
  frame_time, total_observing_time, exposure_time = exosim.modules.timeline_generator.run(opt, channel, planet)
  #Generate noise timelines
  exosim.modules.noise.run(opt, channel, frame_time, total_observing_time, exposure_time)
  #Save
  exosim.modules.output.run(opt, channel, planet)
  
  return star, planet, zodi, channel

if __name__ == "__main__":
  
  xmlFileNameDefault = 'exosim_ariel_mcr_Euro.xml' 
  
  xmlFileName = sys.argv[1]  if len(sys.argv)>1 else xmlFileNameDefault
  
  exosim_msg('Reading options from file ... \n')
  opt = exosim.Options(filename=xmlFileName).opt #, default_path = exosim.__path__[0]).opt
  
  # modify_opt(opt)
  
  star, planet, zodi, channel = run_exosim(opt)
 