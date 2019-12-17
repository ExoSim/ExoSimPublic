import os
import numpy as np
from ..classes import sed
from ..lib import occultquad 
from ..lib.exolib import exosim_error
from ..lib.exolib import exosim_msg
import quantities as aq
import multiprocessing
from pytransit import QuadraticModel

class Planet(object):
  """
    Instantiate a Planet class 
    
    Attributes
    ----------
    wl                          array
                                wavelength [micron]
    sed                         array
                                contains planet-star contrast ratio
  """
  t14 = 0.0
  phi = None
  z   = None
  lc  = None
  
  ldCoeffs = None

  def __init__(self, planet, contrast_path=None, limb_darkening_path = None):
    """
    Initialize a new Planet class 
    
    Parameters
    __________
    contrast_path               string
                                path name of file containg the planet-star contrast spectrum
    """
    self._sanitize_planet(planet)
    self.planet = planet
    self.ldCoeffs = LimbDarkeningCoeffs(limb_darkening_path)
    self.u = None #ld Coeffs interpolated to wl grid
    
    if contrast_path:
      pl_wl, pl_sed = self.read_planet_spectrum(contrast_path)
      idx = np.argsort(pl_wl)      
      self.cr = sed.Sed(pl_wl[idx], pl_sed[idx])  
    else:
      self.cr = None
  
  def _sanitize_planet(self, planet):
    """
    Check that certain expected planet parameters exist and if not 
    add them with a default value
    """
    params_check_dict =  \
               {
                 'eccentricity':0,
                 'inclination':90.0 *aq.deg
               }
    for param in params_check_dict.keys():
      if param not in planet.params.keys():
        planet.params[param] = params_check_dict[param]
      
  def get_t14(self, inc, a, period, planet_radius, star_radius):
    """ t14
    Calculates the transit time 
    
    Parameters
    __________
    inc:                        scalar
                                Planet oprbital inclination [rad]
    a:                          scalar
                                Semimajor axis [meters]
    period:                     scalar
                                Orbital period [seconds]
    planet_radius       :       scalar
                                Planet radius [meters]
    star_radius         :               scalar
                                Star radius [meters]
    
    Returns
    __________
    transit duration : float
        Returns the transit duration [seconds]
    
    Notes
    _____
    Seager, S., & Mallen-Ornelas, G. 2003, ApJ, 585, 1038
      
    
    """
    impact_parameter = np.cos(inc)*a/star_radius
    dtmp = 1+planet_radius/star_radius
    if impact_parameter < dtmp:
      self.t14 = period/np.pi * \
            star_radius/a * \
            np.sqrt(dtmp**2 - impact_parameter**2)
    else:
      #print "WARNING: planet not transiting"
      self.t14 = np.nan
    return self.t14
    
  def read_planet_spectrum(self, contrast_path):
    try:
      _d = np.loadtxt(contrast_path)
    except IOError:
      exosim_error( 'problem reading '+ contrast_path) 

    return _d[:,0]*aq.um, _d[:,1]*aq.dimensionless
  
  def get_orbital_phase(self, t14, period, N=1000):
    f = t14/period
    self.phi = np.linspace(-f.item(), f.item(), N)*f.units
    return self.phi
    
  def eccentric(self, phi, inc, ecc, omega, a, period, star_radius):
    """ eccentric
    Implements an eccentric orbit and calculates the projectedseparation (z)
    
    Parameters
    __________
    phi :       array
      orbital phase 
    inc :       float
      orbital inclination [radiants]
    ecc :       float
      orbital eccentricity [radiants]
    omega :     float
      argument of periastron [radiants]
    a   :       float
      semimajor axis [meters]
    star_radius :       float
      star radius [meters]
    
    Returns
    _______
    z   :       array
      orbital separations
    
    Notes
    _____
    This implementation written by Ingo Waldman 2012
    """
    theta = 2.0 * np.pi * phi 
    aR    = a / star_radius
    
    if ecc < 1e-5:
      # calculating z for a circular orbit
      self.z = aR * np.sqrt(1-((np.cos(theta))**2.*(np.sin(inc))**2.))
      return self.z
    
    # calculating z for eccentric orbits
    n = len(theta)
    E = np.zeros(n)
    ecc2 = np.sqrt((1.+ecc)/(1.-ecc))
    fref = (np.pi / 2.) - omega #setting reference point for the true anomaly
    
    Eref = 2. * np.arctan(1./ecc2 * np.tan(fref/2.)) #reference point for eccentric anomaly
    if Eref < (-np.pi/2.):
      Eref = Eref + 2. * np.pi
      
    Mref = Eref - (ecc * np.sin(Eref)) #reference point for mean anomaly
   
    for i in range(n): 
      # calculating eccentric anomaly using Newton-Rhapson method
      Mtmp = theta[i] + Mref #calculating mean anomaly for phase point
      Etmp = Mtmp
      for j in range(10):
        Etmp = Etmp + ((Mtmp + ecc*np.sin(Etmp) - Etmp) / (1.-ecc*np.cos(Etmp)))
      E[i] = Etmp
    
      # calculating true anomaly
      f = 2.*np.arctan(ecc2*np.tan(E/2.))
      # calculating distance from true anomaly as fraction
      r_frac = (1.-ecc**2.)/(1. + ecc*np.cos(f))
      # computing z
      self.z = 1. - (np.sin(inc)**2.*np.sin(f+omega)**2.)
      self.z = aR*r_frac*np.sqrt(self.z)

    return self.z
    
  #def get_light_curve(self, planet_sed, obs_time, obs_num, transit_type):
  #  return  occultquad(z, u1, u2, p0)[0 if primary_transit else 1] 
    
  def get_light_curve(self, planet_sed, wavelength, timegrid, t0, trinsit_is_primary = True, apply_phase_curve = False):
    """ get_light_curve
    Calculate light curve models based on Mandel & MandelAgol
    
    Parameters
    ----------
    
    planet_sed         : array 
                     the planet CR
    wavelength         : array 
                         wavelength corresponding to the input CR
    timegrid           : array like the timegrid array (units of days) used to generate lightcurves
    t0                 : scalar The time at mid-transit in days
    trinsit_is_primary : boolean True for primary transit
    
    Returns
    -------
    lc         : 2D array 
                 dim=0 contains the wavelength dependence
                 dim=1 contains the time dependence
    z          : array
                 Normalised centre-centre distance
    i0, i1     : scalars
                 index of first and last contact (assuming max contrast ratio)
    
                 
                 
    """
    ##TODO REMOVE useNewCode and entire block when ready
    useNewCode = True
    setLdCoeffsToZero = not(trinsit_is_primary)
    isEclipse = not(trinsit_is_primary)
    
    u =  self.ldCoeffs.getCoeffs(self.planet.star.T, wavelength, forceZero=setLdCoeffsToZero)
    self.u = u ## required for outputing into fits file at the end
    m = QuadraticModel(is_secondary=isEclipse, klims=(0,1))
    m.set_data(timegrid.rescale(aq.day).magnitude)
    
    # m = pytransit.MandelAgol(eclipse=isEclipse )
    # z = m._calculate_z(timegrid.rescale(aq.day).magnitude,
    #             t0.rescale(aq.day), 
    #             self.planet.P.rescale(aq.day), 
    #             self.planet.a/self.planet.star.R.rescale(self.planet.a.units), 
    #             self.planet.i.rescale(aq.radians), 
    #             self.planet.e,
    #             0.0)
    lc = np.zeros( (planet_sed.size, timegrid.size) )
    k2 = (self.planet.R.rescale(aq.m)/ self.planet.a.rescale(aq.m))**2
    albedo = self.planet.albedo
    def apply_mandel_primary(i):
        # lc[i, ...] = m(z, np.sqrt(planet_sed[i]), u[i, ...]) + phase_function * (k2 * albedo) * apply_phase_curve
        lc[i, ...] = m.evaluate_ps(np.sqrt(planet_sed[i]),
                                   u[i, ...],
                                   t0.rescale(aq.day),
                                   self.planet.P.rescale(aq.day),
                                   self.planet.a/self.planet.star.R.rescale(self.planet.a.units),
                                   self.planet.i.rescale(aq.radians),
                                   self.planet.e,
                                   0.0) + phase_function * (k2 * albedo) * apply_phase_curve


    def apply_mandel_secondary(i):
        # f_e = (planet_sed[i] + (m(z, np.sqrt(planet_sed[i]), u[i, ...])-1.0))/planet_sed[i]
        # dtmp = phase_function * (k2 *albedo + planet_sed[i])  if apply_phase_curve else planet_sed[i]
        # lc[i, ...]  = 1.0 + f_e*dtmp

        f_e = (planet_sed[i] + (m.evaluate_ps(np.sqrt(planet_sed[i]),
                                              u[i, ...],
                                              t0.rescale(aq.day),
                                              self.planet.P.rescale(aq.day),
                                              self.planet.a/self.planet.star.R.rescale(self.planet.a.units),
                                              self.planet.i.rescale(aq.radians),
                                              self.planet.e,
                                              0.0)-1.0))/planet_sed[i]

        if apply_phase_curve:
            dtmp = phase_function * (k2 * albedo + planet_sed[i])
        else:
            dtmp = planet_sed[i]

        lc[i, ...] = 1.0 + f_e * dtmp


    if trinsit_is_primary:
        #phaseFactor1 = 1
        #phaseFactor2 = -1
        useMandelFunction = apply_mandel_primary
        phi = np.pi # phase in transit is pi for primari, zero for secondary
    
    else:
        #phaseFactor1 = 0
        #phaseFactor2 = 1
        useMandelFunction = apply_mandel_secondary
        phi = 0.0
            
    alpha = 2*np.pi*( timegrid.rescale(aq.day) - t0.rescale(aq.day) ) / \
                                    self.planet.P.rescale(aq.day) + phi
    ### Old - coded by Andreas
    # phase_function = (phaseFactor1 + phaseFactor2 *  np.cos(alpha))
    
    ### Lambert shpere, Seager's Exoplanet Atmospheres, eq. 3.58
    # phase_function = (np.sin(np.abs(alpha)) + (np.pi-np.abs(alpha))*np.cos(alpha))/np.pi
    
    ### Enzo's simple projection - this is the fraction of the dayside observed
    phase_function = (1+np.cos(alpha))/2.0

    # Calculate z (PyTransit no longer does this)
    # Assumes planet moves at a constant velocity (no eccentricity)
    # Calculate y component (impact parameter)
    z_y = self.planet.a.rescale(self.planet.star.R.units) / self.planet.star.R \
          * np.sin(np.pi/2 - self.planet.i.rescale(aq.radians))
    # Calculate distance of chord traveled
    d   = 2 * (((self.planet.R.rescale(self.planet.star.R.units) + \
                 self.planet.star.R) \
               / self.planet.star.R)**2 - z_y**2)**0.5
    # Velocity of planet across chord
    v   = d / self.t14.rescale(aq.day)
    # X component of z as a function of time
    z_x = v * (timegrid.rescale(aq.day) - t0.rescale(aq.day))
    # Total z vector as a function of time
    z = (z_x**2 + z_y**2)**0.5
    
    #map(useMandelFunction, np.arange(lc.shape[0]) )

    for i in np.arange(lc.shape[0]):
      useMandelFunction(i)
      
    z_12 = 1.0+planet_sed.max()
    idx = np.where(z < z_12)

    return lc, z, idx[0], idx[-1]
    

class LimbDarkeningCoeffs:
  ld_coeffs = None
  def __init__(self, inFile):
    if inFile != None:
      exosim_msg('\nReading limb darkening Coeffs from')
      exosim_msg(inFile)
      self.ld_coeffs = self.parseFile(inFile)
  def parseFile(self, inFile):
    f = open(inFile)
    lines = filter(lambda x:not x.startswith('#') and not len(x)==0, 
          f.read().split('\n'))
    f.close()
    ldCoeffDict = {}
    newCase = False
    for line in lines:
      if line.startswith('CASE'):
        tempLine = line.replace(' ','')
        T_eff = float(tempLine.split('(')[-1].split(',')[0].split('=')[1].replace('K', ''))
        log_g = float(tempLine.split('(')[-1].split(',')[1].split('=')[1].replace(')',''))
        if T_eff in ldCoeffDict.keys():
          raise ValueError('\nlimb-darkening coeffs file contains two entries with the same T_eff')
        ldCoeffDict[T_eff] = {'wl':[], 'gamma1':[], 'gamma2':[]}
        newCase = True
        wl0 = 0
      if newCase and not line.startswith('CASE'):
        temp = line.split(':')
        wl = ( float(temp[0].split(',')[0]) +float(temp[0].split(',')[1])) /2 
        gamma1 = float(temp[1].split(',')[0])
        gamma2 = float(temp[1].split(',')[1])
        if wl<= wl0:
          raise ValueError('\nLimb Darkening wavelength values should be in \
ascending order. Inspect the limb darkening coeffs file.')
        ldCoeffDict[T_eff]['wl'].append(wl)
        ldCoeffDict[T_eff]['gamma1'].append(gamma1)
        ldCoeffDict[T_eff]['gamma2'].append(gamma2)
        wl0 = wl
    return ldCoeffDict
  
  
  def getCoeffs(self, t_eff, wavelegth, forceZero = False):
    gammaOut = np.zeros( (wavelegth.size, 2) )
    if forceZero:
      exosim_msg("\nForcing limb darkening coeffs = 0")
    else:
      if self.ld_coeffs !=None:
        t_eff = t_eff.magnitude
        tempTeff = -1000.
        for tEff in self.ld_coeffs.keys():
          if abs(tEff - t_eff) < abs(tempTeff - t_eff):
            tempTeff = tEff
        ld_model_wl =  np.array(self.ld_coeffs[tempTeff]['wl']) * aq.angstrom
        ld_model_wl = ld_model_wl.rescale(aq.um)
        gamma1 =  np.array(self.ld_coeffs[tempTeff]['gamma1'])
        gamma2 =  np.array(self.ld_coeffs[tempTeff]['gamma2'])
        gamma1_interp =  np.interp(wavelegth, ld_model_wl.magnitude, gamma1)
        gamma2_interp =  np.interp(wavelegth, ld_model_wl.magnitude, gamma2)
        indx = np.where((wavelegth > np.max(ld_model_wl)) | (wavelegth < np.min(ld_model_wl)))
        gamma1_interp[indx] = 0
        gamma2_interp[indx] = 0
        gammaOut = np.vstack((gamma1_interp, gamma2_interp)).T
        exosim_msg("\nInterpolating limbDark coeffs from file")
      else:
        exosim_msg("\nModel not found. Assuming limb darkening coeffs = 0")
    return gammaOut









