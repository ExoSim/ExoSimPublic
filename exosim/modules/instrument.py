from ..classes.sed import Sed
from ..classes.channel import Channel
from ..lib         import exolib

from exosim.lib.exolib import exosim_msg
import time
import numpy           as np
import quantities      as pq
import scipy.constants as spc
import scipy.interpolate

def run(opt, star, planet, zodi):
  
  exosim_msg('Run instrument model ... ')
  st = time.time()
  instrument_emission     = Sed(star.sed.wl, 
                                np.zeros(star.sed.wl.size, dtype=np.float64)* \
                                pq.W/pq.m**2/pq.um/pq.sr)
  instrument_transmission = Sed(star.sed.wl,
                                np.ones(star.sed.wl.size, dtype=np.float64))

  for op in opt.common_optics.optical_surface:
    dtmp_tr = np.loadtxt(op.transmission.replace('__path__', opt.__path__),
                         delimiter=',')
    dtmp_em = np.loadtxt(op.emissivity.replace(  '__path__', opt.__path__),
                         delimiter=',')

    tr = Sed(dtmp_tr[:,0]*pq.um, dtmp_tr[:,1]*pq.dimensionless)
    tr.rebin(opt.common.common_wl)

    em = Sed(dtmp_em[:,0]*pq.um, dtmp_em[:,1]*pq.dimensionless)
    em.rebin(opt.common.common_wl)
    
    exolib.sed_propagation(star.sed, tr)
    exolib.sed_propagation(zodi.sed, tr)
    exolib.sed_propagation(instrument_emission, tr, emissivity=em,
                           temperature=op())
    
    instrument_transmission.sed = instrument_transmission.sed*tr.sed

    
  channel = {}
  for ch in opt.channel:
    #if ch.name != 'NIR Spec': continue
    channel[ch.name] = Channel(star.sed, planet.cr, 
                               zodi.sed, 
                               instrument_emission,   
                               instrument_transmission,
                               options=ch)
    
    ch_optical_surface = ch.optical_surface if isinstance(ch.optical_surface, list) else \
      [ch.optical_surface]
    for op in ch.optical_surface:
      dtmp=np.loadtxt(op.transmission.replace(
          '__path__', opt.__path__), delimiter=',')
      tr = Sed(dtmp[:,0]*pq.um, \
              dtmp[:,1]*pq.dimensionless)
      tr.rebin(opt.common.common_wl)
      em = Sed(dtmp[:,0]*pq.um, \
               dtmp[:,2]*pq.dimensionless)
      em.rebin(opt.common.common_wl)
      exolib.sed_propagation(channel[ch.name].star, tr)
      exolib.sed_propagation(channel[ch.name].zodi, tr)
      exolib.sed_propagation(channel[ch.name].emission, \
              tr, emissivity=em,temperature=op())
      channel[ch.name].transmission.sed *= tr.sed

    # BUG workaround. There is a bug in the binning function. If transmission is zero,
    # it is rebiined to a finite, very small value. This needs to be fixed!
    # For now, I set to zero all transmission smaller than an arbitrary value
    #idx = np.where(channel[ch.name].transmission.sed < 1.0e-5)
    #channel[ch.name].star.sed[idx] = 0.0*channel[ch.name].star.sed.units
    #channel[ch.name].zodi.sed[idx] = 0.0*channel[ch.name].zodi.sed.units
    #channel[ch.name].emission.sed[idx] = 0.0*channel[ch.name].emission.sed.units
    #channel[ch.name].transmission.sed[idx] = 0.0*channel[ch.name].transmission.sed.units
    
    
    # Convert spectral signals
    dtmp=np.loadtxt(ch.qe().replace(
            '__path__', opt.__path__), delimiter=',')
    qe = Sed(dtmp[:,0]*pq.um, \
                 dtmp[:,1]*pq.dimensionless)
    
    Responsivity = qe.sed * qe.wl.rescale(pq.m)/(spc.c * spc.h * pq.m * pq.J)*pq.UnitQuantity('electron', symbol='e-')
    
    Re = scipy.interpolate.interp1d(qe.wl, Responsivity)
    
    Aeff = 0.25*np.pi*opt.common_optics.TelescopeEffectiveDiameter()**2
    Omega_pix = 2.0*np.pi*(1.0-np.cos(np.arctan(0.5/ch.wfno())))*pq.sr
    Apix = ch.detector_pixel.pixel_size()**2
    channel[ch.name].star.sed     *= Aeff             * \
      Re(channel[ch.name].star.wl)*pq.UnitQuantity('electron', 1*pq.counts, symbol='e-')/pq.J
    channel[ch.name].zodi.sed     *= Apix * Omega_pix * \
      Re(channel[ch.name].zodi.wl)*pq.UnitQuantity('electron', 1*pq.counts, symbol='e-')/pq.J
    channel[ch.name].emission.sed *= Apix * Omega_pix * \
      Re(channel[ch.name].emission.wl)*pq.UnitQuantity('electron', 1*pq.counts, symbol='e-')/pq.J
    
    ### create focal plane
    
    #1# allocate focal plane with pixel oversampling such that Nyquist sampling is done correctly 
    fpn = ch.array_geometry()
    fp  = np.zeros( (fpn*ch.osf()).astype(np.int) )

    #2# This is the current sampling interval in the focal plane.  
    fp_delta = ch.detector_pixel.pixel_size() / ch.osf()
    
    
    #3# Load dispersion law 
    if ch.type == 'spectrometer':
      if hasattr(ch, "dispersion"):
        dtmp=np.loadtxt(ch.dispersion.path.replace(
          '__path__', opt.__path__), delimiter=',')
        wav   = dtmp[...,0]
        pathx = dtmp[...,2]*pq.um + ch.dispersion()[0].rescale(pq.um)
        pathy = dtmp[...,3]*pq.um + ch.dispersion()[1].rescale(pq.um)
        pathint = scipy.interpolate.interp1d(pathx, pathy,
                                             bounds_error=False,
                                             fill_value=np.nan,
                                             kind='linear')
        ld = scipy.interpolate.interp1d(pathx, wav, bounds_error=False,
                                        fill_value=0, kind='linear')





      elif hasattr(ch, "ld"):
        # wl = ld[0] + ld[1](x - ld[2]) = ld[1]*x + ld[0]-ldp[1]*ld[2]
        ld = np.poly1d( (ch.ld()[1], ch.ld()[0]-ch.ld()[1]*ch.ld()[2]) )
      else:
        exolib.exosim_error("Dispersion law not defined.")
      
      #4a# Estimate pixel and wavelength coordinates
      x_pix_osr = np.arange(fp.shape[1]) * fp_delta

      if hasattr(ch, "dispersion"):
        y_pix_osr = pathint(x_pix_osr.rescale(pq.um))*pq.um

      elif hasattr(ch, "ld"):
        # middle of focal plane
        y_pix_osr = (np.zeros_like(x_pix_osr.magnitude) + fp.shape[0]//2) * fp_delta

      x_wav_osr = ld(x_pix_osr.rescale(pq.um))*pq.um


      channel[ch.name].wl_solution = x_wav_osr
      channel[ch.name].y_trace     = y_pix_osr


    
    elif ch.type == 'photometer':
      #4b# Estimate pixel and wavelength coordinates
      idx = np.where(channel[ch.name].transmission.sed > channel[ch.name].transmission.sed.max()/np.e)
      x_wav_osr = np.linspace(channel[ch.name].transmission.wl[idx].min().item(),
                              channel[ch.name].transmission.wl[idx].max().item(),
                              8 * ch.osf()) * channel[ch.name].transmission.wl.units
      x_wav_center = (channel[ch.name].transmission.wl[idx]*channel[ch.name].transmission.sed[idx]).sum() / \
        channel[ch.name].transmission.sed[idx].sum()
      
      channel[ch.name].wl_solution = np.repeat(x_wav_center, fp.shape[1])
    
    else:
      exolib.exosim_error("Channel should be either photometer or spectrometer.")
      
    d_x_wav_osr = np.zeros_like(x_wav_osr)
    idx = np.where(x_wav_osr > 0.0)
    d_x_wav_osr[idx] = np.gradient(x_wav_osr[idx])
    if np.any(d_x_wav_osr < 0): d_x_wav_osr *= -1.0
    
    #5# Generate PSFs, one in each detector pixel along spectral axis
    psf = exolib.Psf(x_wav_osr, (y_pix_osr.rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um) * ch.osf().item())\
                     , ch.wfno(), fp_delta, shape='airy')

    # where psf is nans, replace with zeros
    psf[np.isnan(psf)] = 0.0

    #6# Save results in Channel class
    channel[ch.name].fp_delta    = fp_delta
    channel[ch.name].psf         = psf
    channel[ch.name].fp          = fp
    channel[ch.name].osf         = np.int(ch.osf())
    channel[ch.name].offs        = np.int(ch.pix_offs())
    
    channel[ch.name].planet.sed  *= channel[ch.name].star.sed
    channel[ch.name].star.rebin(x_wav_osr)
    channel[ch.name].planet.rebin(x_wav_osr)
    channel[ch.name].zodi.rebin(x_wav_osr)
    channel[ch.name].emission.rebin(x_wav_osr)
    channel[ch.name].transmission.rebin(x_wav_osr)
    channel[ch.name].star.sed     *= d_x_wav_osr
    channel[ch.name].planet.sed   *= d_x_wav_osr
    channel[ch.name].zodi.sed     *= d_x_wav_osr 
    channel[ch.name].emission.sed *= d_x_wav_osr
    
    #7# Populate focal plane with monochromatic PSFs
    if ch.type == 'spectrometer':
      j0 = np.round(np.arange(fp.shape[1]) - psf.shape[1]//2).astype(np.int) 
    
    elif ch.type == 'photometer':
      j0 = np.repeat(fp.shape[1]//2, x_wav_osr.size)
    else:
      exolib.exosim_error("Channel should be either photometer or spectrometer.")
      
    j1 = j0 + psf.shape[1]

    idx = np.where((j0>=0) & (j1 < fp.shape[1]) & (np.isfinite(y_pix_osr)))[0]

    for k in idx:
      i0 = np.int((y_pix_osr[k].rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um)\
                 * ch.osf())//1 - psf.shape[0]//2 + channel[ch.name].offs)
      i1 = np.int(i0 + psf.shape[0])
      channel[ch.name].fp[i0:i1, j0[k]:j1[k]] += psf[...,k] * \
                                   channel[ch.name].star.sed[k]

    #9# Now deal with the planet
    planet_response = np.zeros(fp.shape[1])
    i0p = np.unravel_index(np.argmax(channel[ch.name].psf.sum(axis=2)), channel[ch.name].psf[...,0].shape)[0]
    for k in idx:
      planet_response[j0[k]:j1[k]] += psf[i0p,:,k] * channel[ch.name].planet.sed[k]



    #9# Allocate pixel response function
    kernel, kernel_delta = exolib.PixelResponseFunction(
        channel[ch.name].psf.shape[0:2],
        7*ch.osf(),   # NEED TO CHANGE FACTOR OF 7 
        ch.detector_pixel.pixel_size(),
        lx = ch.detector_pixel.pixel_diffusion_length())

    channel[ch.name].fp = exolib.fast_convolution(
        channel[ch.name].fp, 
        channel[ch.name].fp_delta,
        kernel, kernel_delta)
  
    ## TODO CHANGE THIS: need to convolve planet with pixel response function
    i = 0
    response_temp = np.zeros(fp.shape[1])
    for k in idx:
      i0 = np.int((y_pix_osr[k].rescale(pq.um)/ch.detector_pixel.pixel_size().rescale(pq.um) \
                 * ch.osf().item())//1 - psf.shape[0]//2 + channel[ch.name].offs)
      i1 = np.int(i0 + psf.shape[0])
      response_temp[k] = planet_response[k]/(1e-30+fp[(i0+i1)//2,k])
      i += 1

    channel[ch.name].planet = Sed(channel[ch.name].wl_solution, response_temp)
   
    ## Fix units
    channel[ch.name].fp = channel[ch.name].fp*channel[ch.name].star.sed.units
    channel[ch.name].planet.sed = channel[ch.name].planet.sed*pq.dimensionless
    
    ## Deal with diffuse radiation
    if ch.type == 'spectrometer':
      channel[ch.name].zodi.sed     = scipy.convolve(channel[ch.name].zodi.sed, 
                      np.ones(np.int(ch.slit_width()*channel[ch.name].opt.osf())), 
                      'same') * channel[ch.name].zodi.sed.units
      channel[ch.name].emission.sed = scipy.convolve(channel[ch.name].emission.sed, 
                      np.ones(np.int(ch.slit_width()*channel[ch.name].opt.osf())), 
                      'same') * channel[ch.name].emission.sed.units
    elif ch.type == 'photometer':
      channel[ch.name].zodi.sed = np.repeat(channel[ch.name].zodi.sed.sum(),
                                            channel[ch.name].wl_solution.size)
      channel[ch.name].zodi.wl = channel[ch.name].wl_solution
      channel[ch.name].emission.sed = np.repeat(channel[ch.name].emission.sed.sum(),
                                                channel[ch.name].wl_solution.size)
      channel[ch.name].emission.wl = channel[ch.name].wl_solution
      
    else:
      exolib.exosim_error("Channel should be either photometer or spectrometer.")
      
  exosim_msg(' - execution time: {:.0f} msec.\n'.format((time.time()-st)*1000.0))
  return channel

  pass
