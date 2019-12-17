#Time line generator module
"""
Created on Wed Mar 11 12:51:06 2015

@author: Subi
"""
import time
import numpy as np
import quantities as pq
from ..lib import exolib
from exosim.lib.exolib import exosim_msg
from exosim.lib.spotsim import SpotSim2

def run(opt, channel, planet):
  exosim_msg('Create signal-only timelines ... ')
  st = time.time()
  # Estimate simulation length. Needs to be in units of hours.
  T14 =   planet.get_t14(planet.planet.i.rescale(pq.rad),
     planet.planet.a.rescale(pq.m), 
     planet.planet.P.rescale(pq.s), 
     planet.planet.R.rescale(pq.m), 
     planet.planet.star.R.rescale(pq.m)).rescale(pq.hour)
  
  total_observing_time = T14*(1.0+opt.timeline.before_transit()+opt.timeline.after_transit())
  time_at_transit      = T14*(0.5+opt.timeline.before_transit())
  frame_time           = 1.0/opt.timeline.frame_rate()    # Frame exposure, CLK
      
      
      
  peak1Max = []
  peak2Max = []
  
  
  ##### FOR TESTING - TO BE DELETED
  ADD_STELAR_VARIABILITY = False
  if ADD_STELAR_VARIABILITY:
    workDir = '/home/alp/Work/Ariel/StarVarModels/'
    modelFile = 'M2V_SED_Timeseries.fits'
    fileName = workDir + modelFile
    wlArr, tArr, lc = loadExternalLightCurve(fileName)
    print('qqqqqqqqqqqqqqqqqqqqqqqq===================',lc.shape)
    
    from scipy import interpolate
    f = interpolate.interp2d(wlArr, tArr, lc, kind='cubic')
    
    #qqqq
  
  ##### TESTING END - TO BE DELETED
  
  
  for key in channel.keys():
    
    # Having exposure_time here will allow to have different integration times 
    # for different focal planes.
    exposure_time   = opt.timeline.exposure_time() # Exposure time
    # Estimate NDR rates
    multiaccum     = opt.timeline.multiaccum()    # Number of NDRs per exposure
    allocated_time = (opt.timeline.nGND()+
          opt.timeline.nNDR0()+
          opt.timeline.nRST()) * frame_time
    NDR_time       = (exposure_time-allocated_time)/(multiaccum-1)
    nNDR           = np.ceil(NDR_time/frame_time).astype(np.int).take(0)
    
    # Estimate the base block of CLK cycles
    base = [opt.timeline.nGND().take(0), opt.timeline.nNDR0().take(0)]
    for x in range(int(multiaccum)-1): base.append(nNDR)
    base.append(opt.timeline.nRST().take(0))
    
    # Recalculate exposure time and estimates how many exposures are needed
    exposure_time = sum(base)*frame_time
    number_of_exposures = np.ceil(
      (total_observing_time/exposure_time).simplified.take(0)).astype(np.int)
    total_observing_time = exposure_time*number_of_exposures
    frame_sequence=np.tile(base, number_of_exposures) # This is Nij
    time_sequence = frame_time * frame_sequence.cumsum() # This is Tij
    
    # Physical time of each NDR
    ndr_time = np.dstack([time_sequence[1+i::len(base)] \
      for i in range(int(multiaccum))]).flatten()*time_sequence.units
    # Number of frames contributing to each NDR
    ndr_sequence = np.dstack([frame_sequence[1+i::len(base)] \
      for i in range(int(multiaccum))]).flatten()
    # CLK counter of each NDR
    ndr_cumulative_sequence = (ndr_time/frame_time).astype(np.int).magnitude

    # Create the noise-less timeline
    channel[key].set_timeline(exposure_time,
            frame_time,
            ndr_time, 
            ndr_sequence,
            ndr_cumulative_sequence)
      
    
    # Apply lightcurve model
    cr    =  channel[key].planet.sed[::channel[key].osf]
    cr_wl =  channel[key].planet.wl[::channel[key].osf]
    
    isPrimaryTransit = True if opt.astroscene.transit_is_primary()=='True' else False
    apply_phase_curve = True if opt.astroscene.apply_phase_curve()=='True' else False

    channel[key].lc, z, i0, i1 = planet.get_light_curve(cr, cr_wl, 
              channel[key].ndr_time, 
              time_at_transit, 
              isPrimaryTransit,
              apply_phase_curve)
    
    
    
    channel[key].ldc = planet.u

    ## ADD StelarVariability Model here
    ## Temporary TEST block
    if ADD_STELAR_VARIABILITY:
      tInterp = channel[key].ndr_time - channel[key].ndr_time[0] 
      wlInterp = cr_wl
      lcCorr = f(wlInterp[::-1], tInterp)[::-1, :]
      print() 
      print('\nCR_WL MIN/MAX', np.min(cr_wl), np.max(cr_wl))
      print('=================================', f(0.93345514, 0))
      print('=================================', f(3, 0))
      
      print('##############################')
      print('##############################')
      print('##  APPLYING STELLAR VAR NOISE')
      print('##############################')
      print('##############################')

      import matplotlib.pyplot as plt
      plt.plot(lcCorr)
      plt.show()
      channel[key].lc  *=lcCorr.T
    
    
    
    
    ## ADD SpotSim LC correction here as one line.
    ## Temporary TEST block
    if False:
      print('QQQQQQQQQQQQQQQ MODDED LD CoEFFS')
      channel[key].ldc = np.zeros_like(channel[key].ldc)
      print('QQQQQQQQQQQQQQQ MODDED LD CoEFFS')
      
      spotsList2 = [
         {'r':3, 'x': 27.65, 'y':23.63},
         {'r':6, 'x':-27.41, 'y':-0.19},
         {'r':6, 'x':-30.41, 'y':23.19},
         {'r':7, 'x': -76.41, 'y':-0.19},
         ]
      spotParams = {
          'CONTspot':800, 'CONTfac':100, 'Q':1.6
          }
      Tstar = float(planet.planet.star.T)
      a = float(planet.planet.a.rescale('m'))
      b = np.cos(np.radians(float(planet.planet.i)))*\
            float(planet.planet.a.rescale('m'))/float(planet.planet.star.R.rescale('m'))
      exosim_msg('\n')
      SM = SpotSim2(spotsList2, spotParams, Tstar, a, b, pix=200)
      lc_spotted = np.ones_like(channel[key].lc)
      for i in range(len(cr_wl)):
        exosim_msg('SpotSim - %s - %d/%d\r'%(key, i+1, len(cr_wl)))
        c1 = channel[key].ldc[i, 0]
        c2 = channel[key].ldc[i, 1]
        if  float(cr[i])>0:
          qqqlc, qqqlc_i = SM.calcLC(float(cr_wl[i]), float(cr[i]), c1, c2, z)
          lc_spotted[i, :] = qqqlc
      
      import matplotlib.pyplot as plt
      
      
      tPlot = channel[key].ndr_time
      for i in range(len(cr_wl)):
        if i%(2*np.round((len(cr)/32))**1)==0 and np.mean(lc_spotted[i, :])<.999999:
          yVals = lc_spotted[i, :]# - channel[key].lc[i, :]
          offset = float(np.mean(yVals[(tPlot>3000) & ((tPlot<3300))]))
          peak1Max.append([cr_wl[i], float(np.max((yVals-offset)[(tPlot>2200) & ((tPlot<2700))]))])
          peak2Max.append([cr_wl[i], float(np.max((yVals-offset)[(tPlot>3400) & ((tPlot<3800))]))])
          
          plt.figure(1)
          plt.plot(tPlot,  yVals, label = '%.2f um'%cr_wl[i])
          #plt.plot(tPlot,  lc_spotted[i, :], label = '%.2f um'%cr_wl[i])
          
          if key.startswith('FGS'):
            break
    
    
    
    
    channel[key].set_z(z)
    
  
  
  exosim_msg(' - execution time: {:.0f} msec.\n'.format(
  (time.time()-st)*1000.0))
  return frame_time, total_observing_time, exposure_time

  
      
   
    
def run_(opt, channel, planet):
  exp_time   = opt.timeline.exposure_time() # Exposure time
  multiaccum = opt.timeline.multiaccum()    # Number of NDRs per exposure
  
  ndr_time   = exp_time / multiaccum        # NDR integration time
  
  # Estimate simulation length. Needs to be in units of hours.
  T14 =   planet.get_t14(planet.planet.i.rescale(pq.rad),
     planet.planet.a.rescale(pq.m), 
     planet.planet.P.rescale(pq.s), 
     planet.planet.R.rescale(pq.m), 
     planet.planet.star.R.rescale(pq.m)).rescale(pq.hour)
  
  total_observing_time = T14*(1.0+opt.timeline.before_transit()+opt.timeline.after_transit())
  time_at_transit      = T14*(0.5+opt.timeline.before_transit())
  
  exp_num = np.int(1.0 + total_observing_time / exp_time.rescale(total_observing_time.units)) # number of full exposures to cover the obs time  
  ndr_num = exp_num * multiaccum # Total number of NDRs
  
  total_observing_time = ndr_num * ndr_time.rescale(pq.hour) # make total observing time an integer number
  lc_time = np.arange(ndr_num) * ndr_time.rescale(pq.hour)   # of NDRs
  delta_t_ndr = (lc_time[1]-lc_time[0]).rescale(pq.s)

        
  
    
  for key in channel.keys():
      
    fp = channel[key].fp[channel[key].offs::channel[key].osf, channel[key].offs::channel[key].osf ]
    channel[key].set_timeline(
      np.tile(fp*delta_t_ndr, (lc_time.size, 1, 1)), delta_t_ndr)
    
    cr = channel[key].planet.sed[::channel[key].osf]
    cr_wl =  channel[key].star.wl[::channel[key].osf]
    
    isPrimaryTransit = False
    if opt.astroscene.transit_is_primary()=='True':
      isPrimaryTransit = True
    lc, z, i0, i1 = planet.get_light_curve(cr, cr_wl, lc_time, time_at_transit, isPrimaryTransit)
    channel[key].tl = (channel[key].tl.transpose([1, 2, 0])*lc).transpose([2, 0, 1])
  
  
      
  return lc_time, z

##### FOR TESTING - TO BE DELETED
import astropy.io.fits as pyfits
def loadExternalLightCurve(inFile):
  hdulist = pyfits.open(inFile)
  wlRef = hdulist[0].header['CRVAL1']
  wlDelt = hdulist[0].header['CDELT1']
  timeRef = hdulist[0].header['CRVAL2']
  timeDelt = hdulist[0].header['CDELT2']
  nTimes = hdulist[1].header['TFIELDS']
  nWl =  hdulist[1].header['NAXIS2']
  lc = np.zeros((nWl, nTimes))
  wlArr = np.arange(nWl)*wlDelt+wlRef
  tArr = np.arange(nTimes)*timeDelt+timeRef -100
  print(wlArr)
  print('***** MIN-MAX WL:', np.min(wlArr),np.max(wlArr))
  print('***** MIN-MAX TIME:', np.min(tArr),np.max(tArr))
  for i in range(nTimes):
    if int(i/float(nTimes)*1000)%100==0:
      print("%.1f%%"%(i/float(nTimes)*100))
    lc[:,i] = hdulist[1].data.field('Time %d'%i)

  lc = lc.T/np.mean(lc, 1)
  lc[np.isnan(lc)]=1
  
  return wlArr, tArr, lc



if __name__ == "__main__":
  
  exolib.exosim_error("This module not made to run stand alone")
    
    
