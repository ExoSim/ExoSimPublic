import numpy as np
import quantities as pq
import multiprocessing as mp
from astropy.io import fits
import os, glob, time

from ..lib import exolib
from exosim.lib.exolib import exosim_msg

def run(opt, channel, planet):
  exosim_msg('Save to file ... ')
  st = time.time()
  out_path = os.path.expanduser(
    opt.common.ExoSimOutputPath().replace('__path__', opt.__path__))
  
  if not os.path.exists(out_path):
    os.mkdir(out_path)
  
  existing_sims = glob.glob(os.path.join(out_path, 'sim_*'))
  
  if not existing_sims:
    # Empty list
    sim_number = 0
  else:
    sim_number = sorted([np.int(l.split('_')[-1]) for l in existing_sims])[-1]
    sim_number += 1
  
  out_path = os.path.join(out_path, 'sim_{:04d}'.format(sim_number))
  os.mkdir(out_path)
  
  
    
  for key in channel.keys():
    n_ndr      = channel[key].tl_shape[-1]
    multiaccum = np.int(opt.timeline.multiaccum())
    n_exp      = n_ndr // multiaccum
    hdu        = fits.PrimaryHDU()
    hdu.header['NEXP'] = (n_exp, 'Number of exposures')
    hdu.header['MACCUM'] = (multiaccum, 'Multiaccum')
    hdu.header['TEXP'] = (channel[key].exposure_time.item(), 'Exp Time [s]')
    hdu.header['PLANET'] = (planet.planet.name, 'Planet name')
    hdu.header['STAR'] = (planet.planet.star.name, 'Star name')
    
    #### Detector Simulation Values
    hdu.header['POINTRMS'] = (opt.aocs.pointing_rms.val.item(), 'Model Pointing RMS')
    tempVal = filter(lambda x:x.name==key, opt.channel)[0].detector_pixel.Idc.val.magnitude.item()
    hdu.header['DET_I_DC'] = (tempVal, 'Detector dark current')
    tempVal = filter(lambda x:x.name==key, opt.channel)[0].detector_pixel.sigma_ro.val.magnitude.item()
    hdu.header['DETROERR'] = (tempVal, 'Detector readout noise in e-rms')
    tempVal = filter(lambda x:x.name==key, opt.channel)[0].plate_scale.val.rescale('degree').magnitude.item()
    hdu.header['CDELT1'] = (tempVal, 'Degrees/pixel')
    hdu.header['CDELT2'] = (tempVal, 'Degrees/pixel')
    hdulist = fits.HDUList(hdu)
    
    
    
    
    for i in xrange(channel[key].tl_shape[-1]):
      hdu = fits.ImageHDU(channel[key].noise[..., i].astype(np.float32), name = 'NOISE')
      hdu.header['EXPNUM'] = (i//multiaccum, 'Exposure Number')
      hdu.header['ENDRNUM'] = (i%multiaccum, 'NDR Number')
      hdu.header['EXTNAME'] = 'NOISE'   
      hdulist.append(hdu)
    
    		     
    # Create column data
    col1 = fits.Column(name='Wavelength {:s}'.format(channel[key].wl_solution.units), format='E', 
		       array=channel[key].wl_solution[channel[key].offs::channel[key].osf])
    col2 = fits.Column(name='Input Contrast Ratio', format='E', 
		       array=channel[key].planet.sed[channel[key].offs::channel[key].osf])
    col3 = fits.Column(name='Stellar SED', format='E',
                       array=channel[key].star.sed[channel[key].offs::channel[key].osf])
    cols = fits.ColDefs([col1, col2, col3])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'INPUTS'
    hdulist.append(tbhdu)
    ########
    hdu = fits.ImageHDU(channel[key].outputPointingTl, name = 'SIM_POINTING')
    hdulist.append(hdu)
    ##############
    hdu = fits.ImageHDU(channel[key].ldc, name = 'LD_COEFF')
    hdulist.append(hdu)
    
    ############
    col1 = fits.Column(name='Time {:s}'.format(channel[key].ndr_time.units), format='E', 
		       array=channel[key].ndr_time)
    col2 = fits.Column(name='z', format='E', 
		       array=channel[key].z)
    
    cols = fits.ColDefs([col1, col2])
    tbhdu = fits.BinTableHDU.from_columns(cols)
    tbhdu.name = 'TIME'
    hdulist.append(tbhdu)
    #########
    
    hdu = fits.ImageHDU(channel[key].lc, name = 'LIGHT_CURVES')
    hdulist.append(hdu)
    
    #print hdulist
    hdulist.writeto(os.path.join(out_path, '{:s}_signal.fits'.format(key)))
  
  exosim_msg(' - execution time: {:.0f} msec.\n'.format((time.time()-st)*1000.0))

if __name__ == "__main__":
  
  exolib.exosim_error("This module not made to run stand alone")
