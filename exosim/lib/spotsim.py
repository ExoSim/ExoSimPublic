import numpy as np
import matplotlib.pyplot as plt
from photutils import CircularAperture as CircAp
from photutils import aperture_photometry as ApPhot
import scipy.constants as cnst
from scipy.misc import imrotate
from scipy.ndimage.interpolation import shift
from scipy import signal
import numpy.random as rnd
import time


#==============================================================================
# Primary Transit Class:
#==============================================================================



class SpotSim2():
  def __init__(self, spotsList, spotParams, Tstar, a, b, pix=200):
    """
    PTC: Primary Transit Class
    
    :INPUTS:
    spotsList  -- List containing dicrtiories of spot radius/position (deg)
    spotPrams  -- Dictionary containing:
      CONTspot   -- spot-to-photosphere temperature contrast /kelvin      (scalar)
      CONTfac    -- facula-to-photosphere temperature contrast /kelvin    (scalar)
      Q          -- facula-to-spot area ratio                             (scalar)
    Tstar      -- stellar photosphere temperature /kelvin               (scalar)
    a          -- orbital semi-major axis /metres                       (scalar)
    b          -- planetary impact parameter /stellar radii             (scalar)
    pix        -- number of pixels per star radius                      (integer)
    
    :NOTES:
    v1.0. Built  for implementation in ExoSim stellar variability module (Sarkar 
    & Pascale, 2015) and for independent investigation into stellar variability 
    as part of PX4310 dissertation 'TWINKLE: a British space mission to explore
    faraway worlds'. Luke Johnson (C1216542). Last edit: 17 April 2016.
    
   """
    
    self.Tstar = Tstar
    self.RstarPx = pix
    self.ImHalfSizePx = pix*3/2
    self.spotsList = spotsList
    self.spotParams = spotParams
    self.b = b
    
    
    self.spotsLayerMask = None
    self.spotMasksList = None
    self.initSpotsMask()
    
    
  def planck(self, w1, T):
    a = 1.191042768e8    #*pq.um**5 *pq.W/ pq.m**2 /pq.sr/pq.um
    b = 14387.7516       #*1*pq.um * 1*pq.K
    x = b/(w1*T)
    bb = a/w1**5 / (np.exp(x) - 1.0)
    return bb
  
  def calcSpotMask(self, Rspot, Xspot, Yspot, RstarPx, ImHalfSizePx, Q):
    R  = RstarPx # radius of star in pixels
    r = np.sqrt(Xspot**2 + Yspot**2) # radial distance of spot from centre in pixels
    a = 1.0   
    b = a*np.sin(np.arccos(r/R)) # change in width of ellipse due to projection
   
    Xspot0 = Xspot  # find rotation angle theta and prevent bugs due to 0 vals
    Yspot0 = Yspot
    if Xspot == 0:
        Xspot0 = 1e-3
    if Yspot ==0:
        Yspot0 = 1e-3
    if Xspot == 0 and Yspot == 0:
        theta = 0
    else:
        theta = np.arctan(Xspot0*1.0/Yspot0)*360./(2*np.pi)
   
    y_s = np.linspace(-ImHalfSizePx, ImHalfSizePx, 2*ImHalfSizePx+1) #set up ellipse
    x_s = np.linspace(-ImHalfSizePx, ImHalfSizePx, 2*ImHalfSizePx+1) 
    xx_s, yy_s = np.meshgrid(x_s, y_s)
    arg_s = np.sqrt((xx_s/a)**2 + (yy_s/b)**2)
    R_fac = Rspot*np.sqrt(Q + 1)
    
    spot = np.where(arg_s> Rspot, 0.0, 1.0)
    
    fac = np.where((arg_s> R_fac), 0.0, 2.0)
    spot0 = imrotate(spot, theta,interp='bilinear') # rotate spot and facula 
    fac0 = imrotate(fac, theta,interp='bilinear')               
    spot = np.where(spot0==0, spot0, spot.max())# account for artifact from rotating
    fac = np.where(fac0==0, fac0, fac.max())
    return {'spot':spot, 'fac':fac}
  
  
  def star(self, Num_phot, Rimg, pix):
    F_star = Num_phot
    x = np.linspace(-Rimg, Rimg, 2*Rimg+1, dtype=np.float)
    yy, xx = np.meshgrid(x, x)
    arg = np.sqrt(yy**2 + xx**2)
    arg[arg.shape[0]/2, arg.shape[1]/2] = 1
    arg = np.where(arg< pix, arg, 0)
    arg = np.where(arg<= 0, arg, 1)
    argI = arg*F_star
    return argI
  
  def calcSpotsLayerMask(self, spotsList, spotMaskList):
    
    spotsLayerFac  = np.ones( (2*self.ImHalfSizePx+1, 2*self.ImHalfSizePx+1))
    spotsLayerSpot = np.ones( (2*self.ImHalfSizePx+1, 2*self.ImHalfSizePx+1))
    spotsLayerMask  = np.zeros( (2*self.ImHalfSizePx+1, 2*self.ImHalfSizePx+1), dtype = int)
    for i in range(len(spotsList)):
      spotPosPx = {
          'x':np.round(spotsList[i]['x']*self.RstarPx/90),
          'y':np.round(spotsList[i]['y']*self.RstarPx/90)
        }
      
      # Convert Masks to 0 where spot, 1 else and shift. 
      # This can be done properly in the createMasks(), keeping it refactoring purposes now. 
      spotsLayerSpot *= shift( np.where(spotMaskList[i]['spot']>0, -1, spotMaskList[i]['spot'])+1, 
                              (spotPosPx['y'], spotPosPx['x']), cval = 1)
      spotsLayerFac  *= shift( np.where(spotMaskList[i]['fac']>0, -1, spotMaskList[i]['fac'])+1, 
                              (spotPosPx['y'], spotPosPx['x']), cval = 1)
      
      # Following rounding very important
      spotsLayerSpot = np.round(spotsLayerSpot)
      spotsLayerFac = np.round(spotsLayerFac)
      
    spotsLayerMask = np.where(spotsLayerFac <1, 1, spotsLayerMask)
    spotsLayerMask = np.where(spotsLayerSpot<1, 2, spotsLayerMask)
    
    return spotsLayerMask

  def limb_darkening(self, I0, c1, c2, Rimg, pix):
      x = np.linspace(-1, 1, 2*Rimg+1)*Rimg/pix
      y = np.linspace(-1, 1, 2*Rimg+1)*Rimg/pix
      xx, yy = np.meshgrid(x, y)
      arg = np.sqrt(xx**2 + yy**2)
      u = np.sqrt(1 - arg**2)
      arg_l = I0 * ( 1 - c1*(1-u) - c2*(1-u)**2)
      arg_l = np.where(np.isnan(arg_l), 0, arg_l)
      return arg_l

  def initSpotsMask(self):
    self.spotMasksList = []
    for i in range(len(self.spotsList)):
      self.spotMasksList.append(self.calcSpotMask( self.spotsList[i]['r']*
            self.RstarPx/90, self.spotsList[i]['x']*self.RstarPx/90, 
            self.spotsList[i]['y']*self.RstarPx/90, 
            self.RstarPx, self.ImHalfSizePx, self.spotParams['Q']))
    
    self.spotsLayerMask = self.calcSpotsLayerMask(self.spotsList, self.spotMasksList)
    

  def calcStellarModel(self, wl, c1, c2):
    bb_star = self.planck(wl, self.Tstar)
    Tspot = self.Tstar - self.spotParams['CONTspot']
    Tfac  = self.Tstar + self.spotParams['CONTfac']
    bb_sp = self.planck(wl, Tspot)
    bb_fa = self.planck(wl, Tfac)
    
    star = self.star(bb_star, self.ImHalfSizePx, self.RstarPx)
    spots = np.where(self.spotsLayerMask == 1, bb_fa - bb_star, self.spotsLayerMask)
    spots = np.where(self.spotsLayerMask == 2, bb_sp - bb_star, spots)
    
    
    limb = self.limb_darkening(I0 = 1, c1 = c1, c2 = c2, Rimg = self.ImHalfSizePx, pix = self.RstarPx)
    sim = limb*(star+spots)
    sim_IDEAL = limb*star
    
    return sim, sim_IDEAL
  
  def calcLC(self, wl, cr, c1, c2, zArr):
    zArr_sign = np.zeros_like(zArr)
    zArr_sign[:-1] = np.sign(zArr[1:] - zArr[:-1])
    zArr_sign[-1] = zArr_sign[-2]
    RplanetPx = self.RstarPx * cr**.5
    
    sim, sim_IDEAL = self.calcStellarModel(wl, c1, c2)
    
    star_spotted = np.sum(sim)
    star_IDEAL = np.sum(sim_IDEAL)
    
    lc = np.zeros_like(zArr)
    lc_i = np.zeros_like(zArr)
    for i in range(len(zArr)):
      xPos = zArr_sign[i]*np.sqrt(zArr[i]**2 - self.b**2)*self.RstarPx + self.ImHalfSizePx
      yPos = self.ImHalfSizePx + self.b*self.RstarPx
      if xPos>0 and xPos< sim.shape[0]:
        P_pos = (xPos, yPos)
        P_mask = CircAp(P_pos, RplanetPx)
        planet = ApPhot(sim, P_mask)[0][0]
        planet_I = ApPhot(sim_IDEAL, P_mask)[0][0]
        lc[i] = star_spotted - planet
        lc_i[i] = star_IDEAL - planet_I
        
    lc = np.where(lc==0, np.max(lc), lc)
    lc_i = np.where(lc_i==0, np.max(lc_i), lc_i)
    
    lc /=np.nanmax(lc)
    lc_i /=np.nanmax(lc_i)
    
    return lc, lc_i
  
