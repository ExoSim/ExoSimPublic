import time
import numpy as np
import quantities as pq
from numba import njit

from exosim.lib.exolib import exosim_msg
from ..lib import exolib

@njit
def c_create_jitter_noise(fp, osf, 
                          ndr_sequence, ndr_cumulative_sequence, frame_osf,
                          x_jit, y_jit, 
                          x_offs = 0, y_offs = 0):
  """
  c_create_timeline: create a timeline cube
  
  Parameters
  ----------
  fp     : 2D Array
             The oversampled focal plane
  osf    : scalar
             The oversample factor applied. Relative to physical focal plane
  ndr_sequence: Array
           number of frames in each NDR
  ndr_cumulative_sequence: Array
           number of frames since simulation start corresponding to each NDR 
  frame_osf: scalar
          number of samples in jitter timeline for each frame.
  x_jit  : Array like 
            Jitter in the spectral direction
            Has to be a ndarray of type np.int64
            with shape (nndr, nsndr) where nndr is the number of desired ndr_offs
            and sndr is the number of desired sub_ndr for RPE sampling.
  y_jit  : Array like
            Similar to x_jit, but contains jitter in the spatial direction
  x_offs : Scalar
            Offset in fp in the spectral direction
  y_offs : Scalar
            Offset in fp in the spatial direction
  
  Returns
  -------
  time_line: 3D Array
              The timeline cube of shape (nndr, Nx, Ny) where Nx and Ny are
              fp.shape/osf. The average is returned in each NDR.
  """
  
  # Here time is first axis for memory access efficiency. Will be transposed before return
  time_line = np.zeros( (ndr_sequence.size,
                         np.int(fp.shape[0]//osf),
                         np.int(fp.shape[1]//osf) ) ).astype(np.float32)
  index =   frame_osf*ndr_cumulative_sequence
  nclk  = frame_osf*ndr_sequence


  for ndr in range(index.shape[0]):
    
    j = y_offs
    for y in range(time_line.shape[1]):
      
      i = x_offs
      for x in range(time_line.shape[2]):
        
        for idx in range(index[ndr]-(nclk[ndr]-1), index[ndr]+1):

          j_jit = y_jit[idx] + j
          i_jit = x_jit[idx] + i

          if (i_jit < 0):
            i_jit += fp.shape[1]
          elif (i_jit >= fp.shape[1]):
            i_jit -= fp.shape[1]
            
          if (j_jit < 0):
            j_jit += fp.shape[0]
          elif (j_jit >= fp.shape[0]):
            j_jit -= fp.shape[0]

          time_line[ndr][y][x] += fp[j_jit][i_jit]
          
        i += osf
        time_line[ndr][y][x] /= nclk[ndr]

      j += osf
        
  
  ## Disabled since ExoSim  Issue#42
  #######time_line -= fp[x_offs::osf, y_offs::osf]
  return time_line.transpose( (1,2,0) )



def create_jitter_noise(channel, x_jit, y_jit, frame_osf, frame_time, key, opt):
  
  outputPointingTL = create_output_pointing_timeline(x_jit, y_jit, frame_osf, 
                                ndrCumSeq = channel.ndr_cumulative_sequence )
  
  jitter_x = channel.osf*(x_jit/channel.opt.plate_scale()).simplified
  jitter_y = channel.osf*(y_jit/channel.opt.plate_scale()).simplified
    
  fp_units = channel.fp.units
  fp       = channel.fp.magnitude
  osf      = np.int32(channel.osf)
  offs     = np.int32(channel.offs)

  magnification_factor = np.ceil( max(3.0/jitter_x.std(), 3.0/jitter_y.std()) )
  
  if (magnification_factor > 1):
    try:
      mag = np.int(magnification_factor.item()) | 1
    except:
      mag = np.int(magnification_factor) | 1
      
    fp = exolib.oversample(fp, mag)
    
    #### See ExoSim Issue 42, for following. 
#    fp = np.where(fp >= 0.0, fp, 1e-10)

    osf *= mag
    offs = mag*offs + mag//2
    jitter_x *= mag
    jitter_y *= mag
  
  
  if opt.noise.EnableSpatialJitter() != 'True': jitter_y *= 0.0
  if opt.noise.EnableSpectralJitter() != 'True': jitter_x *= 0.0

  jitter_x = np.round(jitter_x)
  jitter_y = np.round(jitter_y)
  noise = np.zeros((int(fp.shape[0]//osf),
                    int(fp.shape[1]//osf),
                    0)).astype(np.float32)
  indxRanges = np.arange(0,7)*channel.tl_shape[2]//6
  for i in range(len(indxRanges)-1):
    startIdx = int(indxRanges[i])
    endIdx   = int(indxRanges[i+1])
    noise = np.append(noise , c_create_jitter_noise(fp.astype(np.float32), 
                                osf.astype(np.int32),
                                channel.ndr_sequence[startIdx:endIdx].astype(np.int32),
                                channel.ndr_cumulative_sequence[startIdx:endIdx].astype(np.int32), 
                                frame_osf.astype(np.int32),
                                jitter_x.magnitude.astype(np.int32), 
                                jitter_y.magnitude.astype(np.int32), 
                                x_offs = offs.astype(np.int32), 
                                y_offs = offs.astype(np.int32)).astype(np.float32), 
                                axis=2)
  
  ## Multiply units to noise in 2 steps, to avoid 
  ##        Quantities memmory inefficiency 
  qq = channel.ndr_sequence* fp_units*frame_time
  noise = noise*qq
  return  noise, outputPointingTL
  

def create_output_pointing_timeline(jitter_x, jitter_y, frame_osf, ndrCumSeq ):
  """
  Returns  an array  containing the high time resolution pointing offset and the corresponding NDR number.  
  Inputs:
    jitter_x  : array containing spectral pointing jitter time-line
    jitter_x  : array containing spatial pointing jitter time-line
    frameOsf  : Oversampling factor of the 2 pointing arrays
    ndrCumSeq : array associating index number of the above arrays  (before 
                oversampling)where a new NDR takes place.
  Returns:
     A 2d array of 3 columns, where 1st column is NDR number, 
     2nd Column is Oversampled Spectral Pointing jitter
     3rd column is Oversampled Spatial Pointing Jitter
     
  """
  
  pointingArray = np.zeros((ndrCumSeq[-1]*frame_osf, 3))
  indxPrev = 0
  for i, indx in enumerate(ndrCumSeq):
    pointingArray[indxPrev*frame_osf:indx*frame_osf, 0] = i
    pointingArray[indxPrev*frame_osf:indx*frame_osf, 1] = jitter_x[indxPrev*frame_osf:indx*frame_osf]
    pointingArray[indxPrev*frame_osf:indx*frame_osf, 2] = jitter_y[indxPrev*frame_osf:indx*frame_osf]
    indxPrev = indx
  
  return  pointingArray
  
  

def channel_noise_estimator(channel, key, yaw_jitter, pitch_jitter, frame_osf, frame_time, opt):

  # Jitter Noise
  jitNoise, outputPointingTL = create_jitter_noise(channel, yaw_jitter, pitch_jitter, frame_osf, frame_time, key, opt)
  noise = jitNoise
  noise *= channel.lc

  noise += channel.zodi.sed[channel.offs::channel.osf].reshape(-1,1) *\
    frame_time * channel.ndr_sequence * channel.tl_units
  noise += channel.emission.sed[channel.offs::channel.osf].reshape(-1,1) *\
    frame_time * channel.ndr_sequence * channel.tl_units

  noise =  np.rollaxis(noise,2,0)
  noise *= channel.opt.qe_rms_matrix
  noise =  np.rollaxis(noise,0,3)

  noise = np.where(noise >= 0.0, noise, 1e-30) 

  noise  += channel.opt.detector_pixel.Idc() * frame_time * channel.ndr_sequence * channel.tl_units
  
  # Shot Noise
  if opt.noise.EnableShotNoise() == 'True':
    indxRanges = np.arange(0,10)*channel.tl_shape[2]//9
    for i in range(len(indxRanges)-1):
      startIdx = np.int(indxRanges[i])
      endIdx   = np.int(indxRanges[i+1])
      noise[:,:,startIdx:endIdx] = np.random.poisson(noise[:,:,startIdx:endIdx])
  
  # Create ramps
  noise_ = np.zeros((noise.shape[0], noise.shape[1], noise.shape[2]))
  for i in range(0, noise.shape[2], np.int(opt.timeline.multiaccum())):
      for j in range(0, np.int(opt.timeline.multiaccum())):
          noise_[..., i+j] =  noise[..., i:i+j+1].sum(axis=2)  
  noise = noise_
  
  if opt.noise.EnableReadoutNoise() == "True":
    indxRanges = np.arange(0,20)*channel.tl_shape[2]//19
    for i in range(len(indxRanges)-1):
      startIdx = np.int(indxRanges[i])
      endIdx   = np.int(indxRanges[i+1])
      useShape = (channel.tl_shape[0], channel.tl_shape[1],  endIdx-startIdx)
      noise[:,:,startIdx:endIdx] += np.random.normal(scale=channel.opt.detector_pixel.sigma_ro(), size=useShape)*channel.tl_units

  return key, noise, outputPointingTL

def run(opt, channel, frame_time, total_observing_time, exposure_time):
  exosim_msg('Create noise timelines ... ')
  st = time.time()
  
  yaw_jitter = pitch_jitter = frame_osf = None
  
  jitter_file = opt.aocs.PointingModel().replace("__path__", opt.__path__) 
  if hasattr(opt.aocs, 'pointing_rms'): 
    jit_rms = opt.aocs.pointing_rms()
  else:
    jit_rms = None
    
  yaw_jitter, pitch_jitter, frame_osf = exolib.pointing_jitter(jitter_file,
                     total_observing_time, frame_time, rms = jit_rms)  
  
  if opt.aocs.pointing_scan_throw()>0:       
       pitch_jitter = exolib.pointing_add_scan(pitch_jitter,
                            scan_throw_arcsec=opt.aocs.pointing_scan_throw(), 
                            frame_time = frame_time, frame_osf = frame_osf, 
                            exposure_time = exposure_time)
    
  for key in channel.keys():
    
    key, noise,  outputPointingTl = channel_noise_estimator(channel[key], key, yaw_jitter, pitch_jitter, frame_osf, frame_time, opt)
    channel[key].noise = noise
    channel[key].outputPointingTl = outputPointingTl
    
  
  exosim_msg(' - execution time: {:.0f} msec.\n'.format(
  (time.time()-st)*1000.0))
  
  
if __name__ == "__main__":
  
  exolib.exosim_error("This module not made to run stand alone")
