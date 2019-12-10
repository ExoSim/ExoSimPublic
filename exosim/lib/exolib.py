import numpy as np
from scipy import signal
from scipy import interpolate
from scipy.integrate import cumtrapz
import scipy.special
import quantities as pq
import sys, os, pyfits


def exosim_error(error_msg):
    sys.stderr.write("Error code: {:s}\n".format(error_msg))
    sys.exit(0)
    
def exosim_msg(msg, prefix = None):
  msg = msg if prefix==None else "[%s]: %s\n"%(prefix, msg)
  sys.stdout.write(msg)
  sys.stdout.flush()
  
def logbin(x, a,  R, xmin=None, xmax=None):
  n = a.size
  imin = 0
  imax = n-1
  
  if xmin == None or xmin < x.min(): xmin = x.min()
  if xmax == None or xmax > x.max(): xmax = x.max()
  
  idx = np.argsort(x)
  xp = x[idx]
  yp = a[idx]
  
  delta_x = xmax/R
  N = 20.0 * (xmax-xmin)/delta_x
  _x = np.linspace(xmin,xmax, N)
  _y = np.interp(_x, xp, yp)
  
  nbins = 1+np.round( np.log(xmax/xmin)/np.log(1.0 + 1.0/R) ).astype(np.int)
  bins = xmin*np.power( (1.0+1.0/R), np.arange(nbins))
  
  slices  = np.searchsorted(_x, bins)
  counts = np.ediff1d(slices)
  
  mean = np.add.reduceat(_y, slices[:-1])/(counts)
  bins = 0.5*(bins[:-1] + bins[1:])
  return bins[:-1], mean[:-1]

def rebin(x, xp, fp):
  ''' Resample a function fp(xp) over the new grid x, rebinning if necessary, 
    otherwise interpolates
    Parameters
    ----------
    x	: 	array like
	New coordinates
    fp 	:	array like
	y-coordinates to be resampled
    xp 	:	array like
	x-coordinates at which fp are sampled
	
    Returns
    -------
    out	: 	array like
	new samples
  
  '''
  
  if (x.units != xp.units):
    print x.units, xp.units
    exosim_error('Units mismatch')
  
  idx = np.where(np.logical_and(xp > 0.9*x.min(), xp < 1.1*x.max()))[0]
  xp = xp[idx]
  fp = fp[idx]
  
  if np.diff(xp).min() < np.diff(x).min():
    # Binning!
    c = cumtrapz(fp, x=xp)*fp.units*xp.units	
    xpc = xp[1:]
        
    delta = np.gradient(x)
    new_c_1 = np.interp(x-0.5*delta, xpc, c, 
                        left=0.0, right=0.0)*c.units
    new_c_2 = np.interp(x+0.5*delta, xpc, c, 
                        left=0.0, right=0.0)*c.units
    new_f = (new_c_2 - new_c_1)/delta
  else:
    # Interpolate !
    new_f = np.interp(x, xp, fp, left=0.0, right=0.0)*fp.units
  
  '''
  import matplotlib.pyplot as plt
  plt.plot(xp, fp, '-')
  plt.plot(x, new_f, '.-')
  plt.show()
  # check
  print np.trapz(new_f, x)
  idx = np.where(np.logical_and(xp>= x.min(), xp <= x.max()))
  print np.trapz(fp[idx], xp[idx])
  '''
  return x, new_f
  
def fast_convolution(im, delta_im, ker, delta_ker):
  """ fast_convolution.
    Convolve an image with a kernel. Image and kernel can be sampled on different
      grids defined.
    
    Parameters
    __________
      im : 			array like
				the image to be convolved
      delta_im :		scalar
				image sampling interval
      ker : 			array like
				the convolution kernel
      delta_ker :		scalar
				Kernel sampling interval
    Returns
    -------
      spectrum:			array like
				the image convolved with the kernel.
  """
  fc_debug = False
  # Fourier transform the kernel
  kerf = (np.fft.rfft2(ker))
  ker_k = [ np.fft.fftfreq(ker.shape[0], d=delta_ker),
	   np.fft.rfftfreq(ker.shape[1], d=delta_ker) ]
  ker_k[0] = np.fft.fftshift(ker_k[0])
  kerf     = np.fft.fftshift(kerf, axes=0)
  
  # Fourier transform the image
  imf  = np.fft.rfft2(im)
  im_k = [ np.fft.fftfreq(im.shape[0], d=delta_im),
	   np.fft.rfftfreq(im.shape[1], d=delta_im) ]
  im_k[0] = np.fft.fftshift(im_k[0])
  imf     = np.fft.fftshift(imf, axes=0)
  
  # Interpolate kernel 
  kerf_r = interpolate.RectBivariateSpline(ker_k[0], ker_k[1],
					   kerf.real)
  kerf_i = interpolate.RectBivariateSpline(ker_k[0], ker_k[1],
					   kerf.imag)
  if (fc_debug):
    pl.plot(ker_k[0], kerf[:, 0].real,'.r')
    pl.plot(ker_k[0], kerf[:, 0].imag,'.g')
    pl.plot(im_k[0], kerf_r(im_k[0], im_k[1])[:, 0],'-r')
    pl.plot(im_k[0], np.abs(imf[:, 0]),'-b')

  # Convolve
  imf = imf * (kerf_r(im_k[0], im_k[1]) + 1j*kerf_i(im_k[0], im_k[1])) 
  
  if (fc_debug):
    pl.plot(im_k[0], np.abs(imf[:, 0]),'-y')

  imf = np.fft.ifftshift(imf, axes=0)
  
  return np.fft.irfft2(imf)*(delta_ker/delta_im)**2

   
def planck(wl, T):
  """ Planck function. 
    
    Parameters
    __________
      wl : 			array
				wavelength [micron]
      T : 			scalar
				Temperature [K]
				Spot temperature [K]
    Returns
    -------
      spectrum:			array
				The Planck spectrum  [W m^-2 sr^-2 micron^-1]
  """
    
  a = np.float64(1.191042768e8)*pq.um**5 *pq.W/ pq.m**2 /pq.sr/pq.um
  b = np.float64(14387.7516)*1*pq.um * 1*pq.K
  try:
    x = b/(wl*T)
    bb = a/wl**5 / (np.exp(x) - 1.0)
  except ArithmeticError:
    bb = np.zeros(np.size(wl))
  return bb
 
 
def sed_propagation(sed, transmission, emissivity=None, temperature = None):
  sed.sed = sed.sed*transmission.sed
  if emissivity and temperature:
    sed.sed = sed.sed + emissivity.sed*planck(sed.wl, temperature)

  return sed
  
def Psf_Interp(zfile, delta_pix, WavRange):
    ''' 
    PSF Interpolation
    Parametes
    ---------
        zfile : string
            input PSF fits file
        Delta : scalar
            Sampling interval in micron
        WavRange : ndarray
            array of wavelengths in micron
    
    Returns
    -------
        PSF interpolated data cube. Area normalised to unity.
        
    '''
    hdulist = pyfits.open(zfile)    
    NAXIS1, NAXIS2 = hdulist[0].header['NAXIS1'], hdulist[0].header['NAXIS2']
    in_ph_size_x, in_ph_size_y = hdulist[0].header['CDELT1']*NAXIS1, hdulist[0].header['CDELT2']*NAXIS2
    num_pix_x, num_pix_y = np.trunc(in_ph_size_x/delta_pix).astype(np.int), np.trunc(in_ph_size_y/delta_pix).astype(np.int)
               
    inwl   = np.zeros(len(hdulist))
    redata = np.zeros((num_pix_y, num_pix_x, len(hdulist)))

    xin = np.linspace(-1.0, 1.0, NAXIS1)
    yin = np.linspace(-1.0, 1.0, NAXIS2)

    xout = np.linspace(-1.0, 1.0, num_pix_x)
    yout = np.linspace(-1.0, 1.0, num_pix_y)

    for i, hdu in enumerate(hdulist):
        inwl[i]   = np.float64(hdu.header['WAV'])        
        f = interpolate.RectBivariateSpline(xin, yin, hdu.data)
	redata[..., i] = f(xout,yout)

        redata[..., i] /= redata[..., i].sum()
    return interpolate.interp1d(inwl, redata, axis=2, bounds_error=False, fill_value=0.0, kind='quadratic')(WavRange)


def Psf(wl, fnum, delta, nzero = 4, shape='airy'):
  '''
  Calculates an Airy Point Spread Function arranged as a data-cube. The spatial axies are 
  0 and 1. The wavelength axis is 2. Each PSF area is normalised to unity.
  
  Parameters
  ----------
  wl	: ndarray [physical dimension of length]
    array of wavelengths at which to calculate the PSF
  fnum : scalar
    Instrument f/number
  delta : scalar
    the increment to use [physical units of length]
  nzero : scalar
    number of Airy zeros. The PSF kernel will be this big. Calculated at wl.max()
  shape : string
    Set to 'airy' for a Airy function,to 'gauss' for a Gaussian
  
  Returns
  ------
  Psf : ndarray
    three dimensional array. Each PSF normalised to unity
  '''
  
  delta = delta.rescale(wl.units)
  Nx = np.round(scipy.special.jn_zeros(1, nzero)[-1]/(2.0*np.pi) * fnum*wl.max()/delta).astype(np.int)
  
  Ny = Nx = np.int(Nx)
  
  
  if shape=='airy':
    d = 1.0/(fnum*(1.0e-30*delta.units+wl))
  elif shape=='gauss':
    sigma = 1.029*fnum*(1.0e-30*delta.units+wl)/np.sqrt(8.0*np.log(2.0))
    d     = 0.5/sigma**2
    
  x = np.linspace(-Nx*delta.item(), Nx*delta.item(), 2*Nx+1)*delta.units
  y = np.linspace(-Ny*delta.item(), Ny*delta.item(), 2*Ny+1)*delta.units
  
  yy, xx = np.meshgrid(y, x)
  
  if shape=='airy':
    arg = 1.0e-20+np.pi*np.multiply.outer(np.sqrt(yy**2 + xx**2), d)
    img   = (scipy.special.j1(arg)/arg)**2
  elif shape=='gauss':
    arg = np.multiply.outer(yy**2 + xx**2, d)
    img = np.exp(-arg)
  
  norm = img.sum(axis=0).sum(axis=0)
  img /= norm
  
  idx = np.where(wl <= 0.0)
  if idx:
    img[..., idx] *= 0.0
  
  return img

  
  
  
def PixelResponseFunction(psf_shape, osf, delta, lx = 1.7*pq.um, ipd = 0.0*pq.um):
  '''
  Estimate the detector pixel response function with the prescription of 
  Barron et al., PASP, 119, 466-475 (2007).
  
  Parameters
  ----------
  psf_shape	: touple of scalars 
		  (ny, nx) defining the PSF size	
  osf		: scalar
		  number of samples in each resolving element. The 
		  final shape of the response function would be shape*osf
  delta 	: scalar
		  Phisical size of the detector pixel in microns
  lx		: scalar
		  diffusion length in microns
  ipd           : scalar
		  distance between two adjacent detector pixels 
		  in microns
		 
  Returns
  -------
  kernel	: 2D array
		  the kernel image
  kernel_delta  : scalar
                  the kernel sampling interval in microns
  '''
  if type(osf) != int: osf = np.int(osf)
  
  lx += 1e-8*pq.um # to avoid problems if user pass lx=0
  lx = lx.rescale(delta.units)
  
  kernel = np.zeros( (psf_shape[0]*osf, psf_shape[1]*osf) )
  kernel_delta = delta/osf
  yc, xc = np.array(kernel.shape) // 2
  yy = (np.arange(kernel.shape[0]) - yc) * kernel_delta
  xx = (np.arange(kernel.shape[1]) - xc) * kernel_delta
  mask_xx = np.where(np.abs(xx) > 0.5*(delta-ipd))
  mask_yy = np.where(np.abs(yy) > 0.5*(delta-ipd))
  xx, yy = np.meshgrid(xx, yy)
  
  kernel = np.arctan(np.tanh( 0.5*( 0.5*delta - xx)/lx )) - \
	   np.arctan(np.tanh( 0.5*(-0.5*delta - xx)/lx ))
	 
	 
  kernel*= np.arctan(np.tanh( 0.5*( 0.5*delta - yy)/lx )) - \
  	   np.arctan(np.tanh( 0.5*(-0.5*delta - yy)/lx )) 
  
  kernel[mask_yy, ...] = 0.0
  kernel[..., mask_xx] = 0.0

  # Normalise the kernel such that the pixel has QE=1
  kernel *= osf**2/kernel.sum()
  kernel = np.roll(kernel, -xc, axis=1)
  kernel = np.roll(kernel, -yc, axis=0)
  
  return kernel, kernel_delta

def pointing_add_scan(pointing_timeline, scan_throw_arcsec, frame_time, frame_osf, exposure_time):  
  ''' Superimpose saw-tooth scan mode to a pointing jitter timeline.
  The period of a scan is equal exposure time
  
  Parameters
  ----------
  pointing_timeline: Quantities Array 
         Poitning timeline (yaw/pitch) in deg
  scan_throw_arcsec: scalar
         The scan throw in units of arcseconds. 
  frame_time: scalar
      detector frame time in units of time
  frame_osf: scalar
      Frame oversampling factor
  exposure_time: scalar
      time for one exposure containing set of NDRs
  Returns
  -------
  pointing_timeline: Quantities Array
      Pointing timeline  updated with saw-tooth scan pointing superimposed. 
  '''

  tArr = np.arange(len(pointing_timeline) +1)*frame_time/frame_osf   
  tArr = tArr[1:]

  saw_tooth = (tArr.magnitude %exposure_time.magnitude /exposure_time.magnitude)    
  saw_tooth = np.where(saw_tooth > 0.0, saw_tooth, 1.0)
  saw_tooth = saw_tooth*  scan_throw_arcsec.rescale(pq.deg)
  pointing_timeline = saw_tooth + pointing_timeline 
        
  return pointing_timeline   
  


def pointing_jitter(jitter_file, total_observing_time, frame_time, rms=None):  
  ''' Estimate pointing jitter timeline
  
  Parameters
  ----------
  jitter_file: string
	       filename containing CSV columns with 
	       frequency [Hz], Yaw PSD [deg**2/Hz], Pitch [deg**2/Hz]
	       If only two columns given, then it is assumed that 
	       the second column is the PSD of radial displacements
  totoal_observing_time: scalar
      total observing time in units of time
  frame_time: scalar
      detector frame time in units of time
  rms: scalar
      renormalisation rms in units of angle
      
  Returns
  -------
  yaw_jit: jitter timeline in units of degrees
  pitch_jit: jitter rimeline in units of degrees
  osf: number of additional samples in each frame_time needed to capture
       the jitter spectral information
  '''
  
  data = np.genfromtxt(jitter_file, delimiter=',')
  psd_freq = data[..., 0]

  if data.shape[1] > 2:
    psd_yaw = data[..., 1]
    psd_pitch = data[..., 2]
  else:
    psd_yaw = data[..., 1]/2
    psd_pitch = psd_yaw
  
  
  # each frame needs to be split such that jitter is Nyquis sampled
  jitter_sps = 2.0*psd_freq.max()/pq.s
  
  osf = np.ceil(frame_time.rescale(pq.s) * jitter_sps).take(0).astype(np.int)
  if osf < 1: osf = 1
  
  number_of_samples_ = np.int(osf*np.ceil(total_observing_time/frame_time).simplified)-10
  
  number_of_samples = 2**np.ceil(np.log2(number_of_samples_))/2+1
  
  freq_nyq = osf*0.5/frame_time.rescale(pq.s).magnitude
  freq = np.linspace(0.0, freq_nyq, number_of_samples)
  
  # Log Interpolation
  #freq_log = np.log(freq + 1.0e-30)
  #psd_freq_log = np.log(psd_freq + 1.0e-30)
  #psd_yaw_log = np.log(psd_yaw + 1.0e-30)
  #psd_pitch_log = np.log(psd_pitch + 1.0e-30)
  #npsd_yaw   = 1.0e-30+np.interp(freq_log, psd_freq_log, psd_yaw_log, 
				 #left=np.log(1.0e-30), right=np.log(1.0e-30))
  #npsd_pitch = 1.0e-30+np.interp(freq_log, psd_freq_log, psd_pitch_log, 
 				 #left=np.log(1.0e-30), right=np.log(1.0e-30))
  #npsd_yaw = np.exp(npsd_yaw)
  #npsd_pitch = np.exp(npsd_pitch)
  
  # Line interpolation: preserves RMS
  npsd_yaw   = 1.0e-30+interpolate.interp1d(psd_freq, psd_yaw, fill_value=0.0, 
					    kind='linear', bounds_error=None)(freq)
  npsd_pitch   = 1.0e-30+interpolate.interp1d(psd_freq, psd_pitch, fill_value=0.0, 
					      kind='linear', bounds_error=None)(freq)
 
  #import matplotlib.pyplot as plt
  #plt.plot(psd_freq, psd_yaw, 'or')
  #plt.plot(freq, npsd_yaw, '.-g')
  #plt.show()
  
  npsd_yaw    = np.sqrt(npsd_yaw   * np.gradient(freq))
  npsd_pitch  = np.sqrt(npsd_pitch * np.gradient(freq))
  
  yaw_jit_re   = np.random.normal(scale=npsd_yaw/2.0)
  yaw_jit_im   = np.random.normal(scale=npsd_yaw/2.0)
  pitch_jit_re = np.random.normal(scale=npsd_pitch/2.0)
  pitch_jit_im = np.random.normal(scale=npsd_pitch/2.0)
  
  pitch_jit_im[0] = pitch_jit_im[-1] = 0.0
  yaw_jit_im[0]   = yaw_jit_im[-1]   = 0.0
  
  norm = 2*(number_of_samples-1)
  
  yaw_jit = norm*np.fft.irfft(yaw_jit_re + 1j * yaw_jit_im)*pq.deg
  pitch_jit = norm*np.fft.irfft(pitch_jit_re + 1j * pitch_jit_im)*pq.deg

  if rms:
    norm = (rms**2/(yaw_jit[number_of_samples_].var()+ pitch_jit[:number_of_samples_].var())).simplified
    yaw_jit *= np.sqrt(norm)
    pitch_jit *= np.sqrt(norm)
  
  if False:
    print np.sqrt((npsd_yaw**2).sum()), np.sqrt((npsd_pitch**2).sum())
    print yaw_jit.rms(), pitch_jit.rms(), np.sqrt(yaw_jit.var()+pitch_jit.var())
    sps = osf/frame_time
    fp, psdp = signal.periodogram(pitch_jit, sps, window='hamming')
    fy, psdy = signal.periodogram(yaw_jit, sps, window='hamming')
    plt.subplot(211)
    plt.plot(fp, psdp, 'b')
    plt.plot(psd_freq, psd_pitch, 'kx')
    plt.yscale('log'); plt.xscale('log')
    plt.subplot(212)
    plt.plot(fy, psdy, 'r')
    plt.plot(psd_freq, psd_yaw, 'kx')
    plt.yscale('log'); plt.xscale('log')
    
    plt.show()
  
  return yaw_jit, pitch_jit, osf

def jitter__remove(jitter_file, obs_time, ndr_time, rms,mode=2):
    
    """
        Jitter
        
        Simulates 2 d jitter (pointing variation) as a timeline of positional offsets
        Uses Herschel jitter timeline as reference.
        
        Inputs:
        
        1)  jitter_file : reference file with jitter observation timeline
        
        2)  obs_time : total observation time in seconds
        
        3)  ndr_time : time for one non-destructive read
        
        4)  rms : rms of the desired jitter in degrees
        
        5)  mode = 1 : one PSD used to obtain jitter in 2 dimensions
        mode = 2 : two PSDs used - one for each orthogonal dimeension
        
        Output:
        
        1) RA jitter time series in radians (xt) in degrees
        
        2) Dec jitter time series in radians (yt) in degrees
        
        3) time: timegrid of the jitter
        
        4) ndr_osf : number of oversamples per ndr
        
        Requirements:
        
        1) jitter_file : file with known jitter timelines (e.g. Herschel data)
        
        """
    
    f = pyfits.open(jitter_file)  # open a FITS file
    tbdata = f[1].data  # assume the first extension is a table
    time = tbdata['Time']
    ra_t = tbdata['RA']
    dec_t = tbdata['Dec']
    
    if len(time)%2 != 0:  # timeline needs to be even number for real fft
        time = time[0:-1]
        ra_t = ra_t[0:-1]
        dec_t = dec_t[0:-1]
    
    ra_t = (ra_t-np.mean(ra_t))*(np.cos(dec_t*np.pi/180))
    dec_t = dec_t-np.mean(dec_t)

    N = np.float(len(time)) # N = no of samples in reference file
    dt = time[1]-time[0] # dt =sampling period of reference jitter
    fs = 1.0/dt # fs = sampling rate = 2B (B = Nyquist frequency)
    df = fs/N
    freq = np.fft.rfftfreq(np.int(N),d=dt)
    
    ndr_osf = int(1+ ndr_time/dt) # ndr_osf = the minimum integer number of osf to nyquist sample jitter
    new_dt = np.float(ndr_time/ndr_osf) #sampling period for ndr_osf (new_dt is always < dt)
    ndr_number = np.int(obs_time/ndr_time) # total number of full NDRs in the total obs time
    N0 = ndr_number *ndr_osf # total samples for this number of NDRs
    
    x = int(1+np.log(N0)/np.log(2))
    new_N = 2**x # total samples recalculated to be a power of 2 for fourier transforms
    # therefore final number of NDRs > number to fit in obs_time
    # final timeline length > obs_time
    
    new_freq = np.fft.rfftfreq(np.int(new_N),d=new_dt)
    new_fs= 1/new_dt # new_fs must be > fs to ensure nyquist sampling of all frequencies in original power spectrum
    
    ra_f = np.fft.rfft(ra_t)/N
    dec_f = np.fft.rfft(dec_t)/N
    
    ra_psd= 2*abs(ra_f)**2/df
    dec_psd = 2*abs(dec_f)**2/df
    
    ra_psd[0]=1e-30
    ra_psd[-1]=ra_psd[-1]/2
    dec_psd[0]=1e-30
    dec_psd[-1]=dec_psd[-1]/2
    
    # smooth the psd
    
    window_size = 10
    window = np.ones(int(window_size))/float(window_size)
    ra_psd = np.convolve(ra_psd, window, 'same')
    dec_psd = np.convolve(dec_psd, window, 'same')
    
    # resample and 'zero pad' to new frequency grid and N
    
    f1 = interpolate.interp1d(freq, ra_psd,bounds_error=False,fill_value=1e-30, kind='linear')
    f2 = interpolate.interp1d(freq, dec_psd,bounds_error=False,fill_value=1e-30, kind='linear')
    
    new_ra_psd = f1(new_freq)
    new_dec_psd = f2(new_freq)
    
    #plt.figure('power spectrum regrid')
    #plt.plot(new_freq, new_ra_psd, freq, ra_psd)
    
    psd = [new_ra_psd,new_dec_psd]
    N = new_N
    fs = new_fs
    df = fs/N
    
    if mode == 1:
        #new to work on how to scale this
        
        comb_t = np.sqrt(ra_t**2 + dec_t**2)
        comb_f = np.fft.rfft(comb_t)/N
        comb_psd = 2*abs(comb_f)**2/df
        comb_psd[0]=1e-30
        comb_psd[-1]=comb_psd[-1]/2
        
        phi_t = np.arctan2(dec_t,ra_t)
        phi_f = np.fft.rfft(phi_t)/N
        phi_psd = 2*abs(phi_f)**2/df
        phi_psd[0]=1e-30
        phi_psd[-1]=phi_psd[-1]/2
        
        psd1 = psd[0]
        ps1 = psd1*df/2
        amp = np.random.normal(0,np.sqrt(ps1),len(ps1))
        phi = np.random.uniform(0,np.pi,len(ps1))
        zf = amp*np.exp(phi*1j)
        zt = np.fft.irfft(zf)*N
        
        psd2 = psd[1]
        ps2 = psd2*df/2
        amp = np.random.normal(0,np.sqrt(ps2),len(ps2))
        phi = np.random.uniform(0,np.pi,len(ps2))
        anglef = amp*np.exp(phi*1j)
        anglet = np.fft.irfft(anglef)*N
        
        xt = zt*np.cos(anglet)
        yt = zt*np.sin(anglet)
    
    
    elif mode == 2:
        
        psd1 = psd[0]
        ps = psd1*df/2
        amp = np.random.normal(0,np.sqrt(ps),len(ps))
        phi = np.random.uniform(0,np.pi,len(ps))
        xf = amp*np.exp(phi*1j)
        xt = np.fft.irfft(xf)*N
        
        psd2 = psd[1]
        ps = psd2*df/2
        amp = np.random.normal(0,np.sqrt(ps),len(ps))
        phi = np.random.uniform(0,np.pi,len(ps))
        yf = amp*np.exp(phi*1j)
        yt = np.fft.irfft(yf)*N
    
    
    else:
        print "error: maximum of 2 psds can be used"
    

    xt = xt[0:N0]   # only need N0 samples to cover the observation
    yt = yt[0:N0]
    
    xt = xt*(rms*(np.sqrt(2)/2)/np.std(xt))  # only need N0 samples to cover the observation
    yt = yt*(rms*(np.sqrt(2)/2)/np.std(yt))

    time = np.arange(0,(N0)*new_dt,new_dt) # timegrid for final jitter timelines

    return xt,yt,time,ndr_osf

def oversample(fp, ad_osf):
    
    xin = np.linspace(0,fp.shape[1]-1,fp.shape[1])
    yin = np.linspace(0,fp.shape[0]-1,fp.shape[0])
    x_step =  abs(xin[1]) - abs(xin[0])
    y_step =  abs(yin[1]) - abs(yin[0])
    
    # calculates the new step sizes for new grid
    x_step_new = np.float(x_step/ad_osf)
    y_step_new = np.float(y_step/ad_osf)
    
    # new grid must start with an exact offset to produce correct number of new points
    x_start = -x_step_new * np.float((ad_osf-1)/2)
    y_start = -y_step_new * np.float((ad_osf-1)/2)
    
    # new grid points- with correct start, end and spacing
    xout = np.arange(x_start, x_start + x_step_new*fp.shape[1]*ad_osf, x_step_new)
    yout = np.arange(y_start, y_start + y_step_new*fp.shape[0]*ad_osf, y_step_new)
    
    # interpolate fp onto new grid
    fn = interpolate.RectBivariateSpline(yin,xin, fp)
    new_fp = fn(yout,xout)
    
    return new_fp


def animate(Data):
    
    from mpl_toolkits.mplot3d import Axes3D
    import matplotlib.pyplot as plt

#    Data = data['channel']['SWIR'].timeline

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    wframe = None
    
    for j in range(0,Data.shape[2]):
    
            oldcol = wframe

            X = np.arange(0, Data.shape[1])
            Y = np.arange(0, Data.shape[0])
            X, Y = np.meshgrid(X, Y)
            
           
        
            Z = Data[...,j]
#            print Z.sum()
            
            wframe = ax.plot_wireframe(X, Y, Z, rstride=2, cstride=2)
            
#            ax.set_zlim(0,20000)
#            ax.set_title(j)
#            
     
        

    
        # Remove old line collection before drawing
            if oldcol is not None:
                ax.collections.remove(oldcol)
    
            plt.pause(.01)