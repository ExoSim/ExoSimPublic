import numpy as np
import scipy.ndimage
import copy, os, pyfits
from ..lib import exolib

class Channel(object):
  
  planet          = None
  star            = None
  zodi            = None
  emission        = None
  transmission    = None
  psf             = None
  fp              = None   # focal plane
  fp_delta        = None   # Focal plane sampling interval (equal for 
			# spatial -y- and spectral -x- directions)
  osf             = 1      # Focal plane oversampling factor
  wl_solution     = None   # This is the wavelength solution 
			#   for the focal plane at its current 
			#   sampling. 
  opt             = None   # Options relevant for this
                        # channel
  offs            = 1      # pixel offset for fp sampling
			# fp[offs::opt.osf()]
  tl_shape        = None #
  tl_units        = None
  tl              = None    # timeline cube
  noise           = None    # Noise cube on timeline
  
  def __init__(self, star, planet, zodi, emission, 
	       transmission, options=None):
    self.star         = copy.deepcopy(star)
    self.planet       = copy.deepcopy(planet)
    self.zodi         = copy.deepcopy(zodi)
    self.emission     = copy.deepcopy(emission)
    self.transmission = copy.deepcopy(transmission)
    self.is_spec = True
    if options : 
      self.opt = options
      if options.type in ['spectrometer', 'photometer']:
        self.instrument_type = options.type
      else:
        raise ValueError('XML "channel" can be either "spectrometer" or "photometer". Current value is "%s"'%options.type)
    
    
  def save(self, pathname=None, sim_num=0, file_ext='fits', planet=None):
    if not pathname: pathname = '.'
    full_path = os.path.expanduser(os.path.join(pathname, str(sim_num), 'static'))
    try:
      os.makedirs(full_path)
    except os.error:
      pass
    
    filename = os.path.join(full_path, self.opt.name)
   
    if file_ext == 'fits':
      prihdr = pyfits.Header()
      prihdr['wavsol_0'] = (self.opt.ld().base[0], 'reference pixel wl')
      prihdr['wavsol_1'] = (self.opt.ld().base[1], '')
      prihdr['wavsol_2'] = (self.opt.ld().base[2], 'reference pixel')
      prihdr['BUNITS']   = "{:>18s}".format(str(self.fp.units))
      if planet:
	prihdr['NAME'] = ("{:>18s}".format(planet.planet.name), '')
	prihdr['T14'] = (float(planet.t14), str(planet.t14.units))
	prihdr['PERIOD'] = (float(planet.planet.P), 
			    str(planet.planet.P.units))
	
      fp_hdu = pyfits.PrimaryHDU(self.fp, header=prihdr)
      tb_hdu = pyfits.new_table(pyfits.ColDefs([
	pyfits.Column(name='wl', format='E', array=self.wl_solution),
	pyfits.Column(name='cr', format='E', array=self.planet.sed),
	pyfits.Column(name='star', format='E', array=self.star.sed)]))
	
      
      hdulist = pyfits.HDUList([fp_hdu, tb_hdu])
      hdulist.writeto(filename + '.' + file_ext, clobber=True)
    else:
      exolib.exosim_error('channel.save - file format not supported')
  
  def set_timeline(self, exposure_time, frame_time, ndr_time, ndr_sequence, ndr_cumulative_sequence):
    '''
    set_timeline: initiate the timeline by repeating the focal plane fp at each ndr_cumulative_sequence
    
    Parameters
    ----------
    frame_time: scalar
		the frame time in second. This is the detector sample rate.
    ndr_time : vector
	       Physical time of each NDR (s)
    ndr_sequence: vector
                  Number of frames contributing to each ndr_cumulative_sequence
    ndr_cumulative_sequence: CLK counter of each NDR
    '''
    
    fp = self.fp[self.offs::self.osf, self.offs::self.osf]
    self.tl_units = fp.units*frame_time.units
    self.tl_shape = (fp.shape[0], fp.shape[1], len(ndr_sequence)) 
    self.exposure_time = exposure_time
    self.ndr_time = ndr_time
    self.ndr_sequence = ndr_sequence
    self.ndr_cumulative_sequence = ndr_cumulative_sequence
    
  def set_noiseline(self, noise):
    self.noise = noise
  
  def set_z(self, z):
    self.z = z
  
  def get_timeline(self):
    return self.tl
  
  def magnify(self, magnification_factor, order=3):
    
    try:
      mag = magnification_factor.item()
    except:
      mag = magnification_factor
      
    self.fp = scipy.ndimage.zoom(self.fp, mag, 
				 order=order)*self.fp.units
    self.wl_solution = scipy.ndimage.zoom(self.wl_solution, mag,
					  order=order)*self.wl_solution.units
    self.osf *= mag
    self.fp_delta /= mag
    
      
    