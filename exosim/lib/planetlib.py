import numpy as np
from exosim import exolib
class PlanetFunctions:
    '''
    A collection of planet related functions
    '''
    @staticmethod
    def planet_spectrum_pri (wl, Tplanet, mu, planet_mass, Rp, Rs):
        '''
        Calculate contrast ratio  for primary planet transit
        
        :param          wl: wavelength array [um]
        :type           wl: numpy.array
        :param     Tplanet: Planet temperature  [K]
        :type      Tplanet: float
        :param          mu: Mean molecular weight [kg]
        :type           mu: float
        :param planet_mass: Planet mass [kg]
        :type  planet_mass: float
        :param          Rp: Planet radius [m]
        :type           Rp: float
        :param          Rs: Star radius [m]
        :type           Rs: float
        :returns          : array like wl -- Contrast ratio of planet primary transit 
        '''
        # k and G are boltzman and Gravitational constants
        G= 6.674e-11
        k= 1.3806488e-23
        planet_gravity = (G * planet_mass) / (Rp**2)
        scale_height = (k * Tplanet) / (mu * planet_gravity)
        planet_sed = np.empty_like(wl)
        planet_sed[:] = (((Rp + 5* scale_height) / Rs) ** 2.) 
        return planet_sed
    
    @staticmethod
    def planet_spectrum_sec (wl, Tstar, Tplanet, Rp, Rs, D, albedo):
        '''
        Calculate contrast ratio  for secondary planet transit
        
        :param          wl: wavelength array [um]
        :type           wl: numpy.array
        :param       Tstar: Star temperature  [K]
        :type        Tstar: float
        :param     Tplanet:  Planet temperature  [K]
        :type      Tplanet: float
        :param          Rp: Planet radius [m]
        :type           Rp: float
        :param          Rs: Star radius [m]
        :type           Rs: float
        :param           D: Mean star-planet distance [m]
        :type            D: float
        :param      albedo: Planet albedo
        :type       albedo: float
        :returns          : array like wl -- Contrast ratio of planet secondary transit 
        '''
        planet_flux = exolib.planck(wl, Tplanet)  * np.pi
        star_flux = exolib.planck(wl, Tstar) * np.pi
        flux_ratio = planet_flux / star_flux * (Rp /Rs) ** 2
        reflected_frac  =   albedo * (Rp/D)**2
        planet_sed = flux_ratio + reflected_frac
        return planet_sed
    
    @staticmethod
    def calcTplanet(Tstar, Rs, D, albedo, tidal_lock):
        '''
        :parameter       Tstar: Star temperature [K]
        :type            Tstar: float
        :param              Rs: Star radius [Solar Radii]
        :type               Rs: float
        :parameter           D: Mean star-planet distance [AU]
        :type                D: float
        :param          albedo: Planet albedo
        :type           albedo: float
        :param      tidal_lock: assume tidal lock in calculation
        :type       tidal_lock: bool
        '''
        # Rs needs to be in units of solar radii for this calculation
        # D in AU
        if tidal_lock == True:
            Tplanet = (1 - albedo)**0.25 * (331 * Tstar / 5770) * (Rs/D)**0.5
        else:
            # for f  = 1
            Tplanet = (1- albedo)**0.25 * (279* Tstar/ 5570) * (Rs/D)**0.5
        return Tplanet