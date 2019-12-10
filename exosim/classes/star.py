import numpy   as np
import quantities as pq
from astropy.io import fits
import os, time, tables, glob, sys
from   ..classes import sed
from   ..lib import exolib

class Star(object):
  """
    Instantiate a Stellar class using Phenix Stellar Models
    
    Attributes
    ----------
    lumiosity : 		float
				Stellar bolometric luminosity computed from Phenix stellar modle. Units [W]
    wl				array
				wavelength [micron]
    sed 			array
				Spectral energy density [W m**-2 micron**-1]
    ph_wl			array
				phoenix wavelength [micron]. Phenix native resolution
    ph_sed 			array
				phenix spectral energy density [W m**-2 micron**-1]. Phenix native resolution
    ph_filename 		phenix filename
  """
  #def __init__(self, star_sed_path, star_distance, star_temperature, star_logg, star_f_h, star_radius):
  def __init__(self, exocat_star, star_sed_path, use_planck_spectrum=False):
    """ 
    Parameters
    __________
      exocat_star 	: 	object
				exodata star object
      star_sed_path:    : 	string
				path to Phoenix stellar spectra
      
    """
    
    ph_wl, ph_sed, ph_L, ph_file = self.read_phenix_spectrum(star_sed_path, 
						    exocat_star.d.rescale(pq.m), 
						    exocat_star.T, 
						    exocat_star.calcLogg(), 
						    exocat_star.Z, 
						    exocat_star.R.rescale(pq.m))
    if use_planck_spectrum:
      ph_wl, ph_sed = self.get_star_spectrum(ph_wl, 
			    exocat_star.d.rescale(pq.m), 
			    exocat_star.T, 
			    exocat_star.R.rescale(pq.m))


    self.ph_luminosity = ph_L
    self.ph_sed        = sed.Sed(ph_wl, ph_sed)
    self.sed           = sed.Sed(ph_wl, ph_sed)
    self.luminosity    = ph_L
    self.ph_filename   = ph_file
          
  def read_phenix_spectrum(self, path, star_distance, star_temperature, star_logg, star_f_h, star_radius):
    """Read a PHENIX Stellar Spectrum. 
    
    Parameters
    __________
      path : 			string
				path to the file containing the stellar SED [erg/s/cm^2/cm]
      star_temperature : 	scalar
				Stellar temperature [K]
      star_logg : 		scalar
				Stellar log_10(g), where g is the surface gravity
      star_F_H : 		scalar
				Stellar metallicity [F/H]
    Returns
    -------
      wl:			array
				The Wavelength at which the SED is sampled. Units are [micron]
      sed :			array
				The SED of the star. Units are [W m**-2 micron**-1]
      L :			scalar
				The bolometric luminosity of the star. Units are [W]
      filename			string
				The phoenix stellar spectrum used
				
    """
    
####################### USING PHOENIX BIN_SPECTRA BINARY FILES (h5)
    
    sed_name = glob.glob(os.path.join(path, "*.BT-Settl.spec.fits.gz"))
    if len(sed_name) == 0:
      exolib.exosim_error("No stellar SED files found")
    
    
    sed_T_list    = np.array( [np.float(os.path.basename(k)[3:8])   for k in sed_name])
    sed_Logg_list = np.array( [np.float(os.path.basename(k)[9:12])  for k in sed_name])
    sed_Z_list    = np.array( [np.float(os.path.basename(k)[13:16]) for k in sed_name])
    
    idx = np.argmin(np.abs(sed_T_list - np.round(np.float(star_temperature)/100.0)) + 
		    np.abs(sed_Logg_list-star_logg) + 
		    np.abs(sed_Z_list - star_f_h))
    
    
    
    ph_file = sed_name[idx]
    with fits.open(ph_file) as hdu:
      wl = pq.Quantity(hdu[1].data.field('Wavelength'),
		     hdu[1].header['TUNIT1']).astype(np.float64)
      sed = pq.Quantity(hdu[1].data.field('Flux'), pq.W/pq.m**2/pq.micron).astype(np.float64)
      if hdu[1].header['TUNIT2'] != 'W / (m2 um)': print 'Exception'
    
      #remove duplicates
      idx = np.nonzero(np.diff(wl))
      wl = wl[idx]
      sed = sed[idx]
      hdu.close()

      
    
    
    # Calculate Luinosity for consistency check
    bolometric_flux        =  np.trapz(sed, x = wl) 			      # [W m**-2]
    bolometric_luminosity  =  4*np.pi * star_radius**2 * bolometric_flux      # [W]
    sed                   *=  (star_radius/star_distance)**2 		      # [W/m^2/mu]
    
    return wl, sed, bolometric_luminosity.rescale(pq.W), ph_file
    
  def get_star_spectrum(self, wl, star_distance, star_temperature, star_radius):
    omega_star = np.pi*(star_radius/star_distance)**2 * pq.sr
    sed = omega_star*exolib.planck(wl, star_temperature)
    return wl, sed


  def get_limbdarkening(self, filename):    
    lddata = np.loadtxt(filename)
    return lddata
    

    
    
    
    
    
    
    ### This block reads BT-Settl text files 
    
    #wl_filename  = os.path.join(path,"lte{:03.0f}-{:01.1f}-{:01.1f}a+0.0.BT-Settl.spec.7.bz2".format(
      #np.round(star_temperature/100.0), star_logg, star_f_h))
    #import string
    #rule = string.maketrans('D', 'E')
    #raw_data = np.loadtxt(wl_filename, usecols=(0,1),
			  #converters={1:lambda val: float(val.translate(rule))})
    #wl  = raw_data[...,0]
    #sed = raw_data[...,1]
    ## Unit conversion
    #wl *= 1.0e-4 # [um]
    #sed = 10**(sed + 8.0 - 15.0) # [W m^-2 mu^-1]
    
    


####################### USING PHOENIX FITS FILES 
    
#    wl_filename  = os.path.join(path,"WAVE_PHOENIX-ACES-AGSS-COND-2011.fits")
#    wl           = pyfits.getdata(wl_filename) # [angstrom]
#    sed_filename = os.path.join(path, 
#      "lte{:05.0f}-{:03.2f}-{:02.1f}.PHOENIX-ACES-AGSS-COND-2011-HiRes.fits".format( np.round(star_temperature/100.0)*100.0, star_logg, star_f_h))
# 
#    sed = pyfits.getdata(sed_filename) # [erg/s/cm^2/cm]
#
#    hdulist      = pyfits.open(sed_filename)
#    radius       = hdulist[0].header['PHXREFF'] # [cm]
#    hdulist.close()
#
#    sed = pyfits.getdata(sed_filename) # [erg/s/cm^2/cm
#      
#    # Unit conversion
#    wl         *= 1.0e-4 # [um]
#    sed        *= 1.0e-7 # [W m^-2 mu^-1]
#    radius     *= 1.0e-2 # [m]
      
