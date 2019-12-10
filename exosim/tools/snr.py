"""
Created on Thu May 14 09:21:01 2015

@author: Subhajit Sarkar
Cardiff University

A Tool that reads in Exosim-created fits files 
and retrieves spectra and SNR from the 'data' output of ExoSim
The spectra are binned on a wavelength grid defined by R-based binning

Basic usage:

from exosim.tools.snr import SnrTool
snrTool = SnrTool()
wl, signal, noise = snrTool.calc_exolib_SNR(fitsFilePath, R, ld, pix_size)
wl, fittedDepth = snrTool.fit_lc_mandel_agol(fitsFilePath, R, ld, pix_size, doCds)

where:
fitsFilePath : pathe to fits file created by ExoSim
R            : R-based binning parameter 
ld           : Wavelength solution 
               (q, m, x_ref): wl = q + m(x - x_ref) where x is the pixel coordinate 
               in units of micron"
pix_size     : pixel size in microns              
"""

import pyfits
import numpy as np
import quantities  as aq
import pytransit
from scipy.optimize import fmin

class SnrTool():
    def __init__(self):
        self.lcf = LightcurveFitting()
        self.fdp = FitsDataProcessing()
    def calc_exosim_SNR(self, fitsFileName, R, ld, pixSize, doCds):
        R = float(R) ## If R is not float, this can lead to wrong results.
        wl, signalBin, noiseBin, bin_pos = self.fdp.bin_spec_data_R(fitsFileName, R, ld, pixSize, doCds)
        signal = np.zeros(signalBin.shape[1])
        noise = np.zeros(signalBin.shape[1])
        nExp = signalBin.shape[0]
        for i in range (signalBin.shape[1]):
            signal[i] = (signalBin[:,i].max() - signalBin[:,i].min()) /signalBin[:,i].max()
            noise[i]  =  (np.std(noiseBin[:,i])    / signalBin[:,i].max()   )/ np.sqrt(nExp/2.)
        
        
        return wl, signal, noise
    
    def fit_lc_mandel_agol(self, fitsFileName, R, ld, pixSize, doCds):
        R = float(R) ## If R is not float, this can lead to wrong results.
        wl, signalBin, noiseBin, bin_pos = self.fdp.bin_spec_data_R(fitsFileName, R, ld, pixSize, doCds)
        hdul = pyfits.open(fitsFileName)
        ##z = hdul[-1].data['z'][1::2]
        multiaccum = hdul[0].header['MACCUM']
        z = hdul[-1].data['z'][multiaccum-1::multiaccum]
        hdul.close()
        u = [0,0] # modify by using u from fit files later
        sig = signalBin*1
        no2 = noiseBin*1
        fittedDepth =  np.zeros(signalBin.shape[1])
        for i in range (signalBin.shape[1]):
            no2[:,i] = no2[:,i] /  sig[:,i].max()
            sig[:,i] = sig[:,i] / sig[:,i].max()
            sig[:,i] = sig[:,i] + no2[:,i]
            fittedDepth[i] = self.lcf.fit_curve_fmin(sig[:,i],z,u)                      
        return wl, fittedDepth




class LightcurveFitting():
    def __init__(self):
        self.modelMandelAgol =  pytransit.MandelAgol(eclipse=False)

    
    
    def fit_curve_fmin(self, lc,z, u):  
        # fits a Mandel & Agol lightcurve using simplex downhill minimisation
        #' returns the fitted Depth
        p_init = 0.01  # planet/star radius ratio initial estimate
        data_std = np.std(lc[0:len(lc)/4])+ 1e-8  # data variance estimate
        p_fit, err, out3, out4, out5 = fmin(self.chi_sq, p_init, args=(data_std, lc, z, u), disp=0, full_output=1)
        depth = p_fit[0]**2 
        return depth

    def chi_sq (self, p, sigma, lc, z, u): 
        # calculates the sum of squares or RESIDUAL = DATA - MODEL
        model = self.modelMandelAgol(z, p, u)
        chi_sq  = np.sum((lc - model) / sigma)**2
        return chi_sq


class FitsDataProcessing():
    def bin_spec_data_R(self, fitsFileName, R, ld, pixSize, doCds):
        R = float(R) ## If R is not float, this can lead to wrong results.
        hdul = pyfits.open(fitsFileName)
        wl0 = hdul[-2].data['Wavelength 1.0 um'][0]
        signal_stack, noise_stack = self.extract_exosim_spectrum(fitsFileName, doCds)
        signalBin, wl, bin_pos = self.div_in2_wav_2(signal_stack, wl0, R, ld, pixSize)
        noiseBin , wl, bin_pos = self.div_in2_wav_2(noise_stack,  wl0, R, ld, pixSize)       
        hdul.close()
        return wl, signalBin, noiseBin, bin_pos

    def extract_exosim_spectrum(self, fitsFilePath, doCds):
        ## Assumes the fits file contains:
        ## 0th HDU: PrimaryHDU
        ## 1th-Nth: ImageHDUs of the same shape
        ## Perhaps a CREATOR:Exolib in fits header meta key
        ## would resolve potential issues
        hdul = pyfits.open(fitsFilePath)
        multiaccum = hdul[0].header['MACCUM']
        nExp = hdul[0].header['NEXP']
        
        signal = np.zeros((hdul[1].shape[0], hdul[1].shape[1], nExp))
        noise = np.zeros((hdul[1].shape[0], hdul[1].shape[1], nExp))
        for i in range (nExp):
            ##
            idx_1 =  i*multiaccum*2   + 1
            idx_2 =  idx_1 + (multiaccum*2 - 2)
            ## FIXME: Its hould be "if doCDS" and if "multiaccum"
            if multiaccum>1:
                if doCds:
                    #if CDS being used.... 
                    cds_sig = hdul[idx_2].data - hdul[idx_1].data 
                    cds_noise = hdul[idx_2 +1].data - hdul[idx_1 +1].data
                else:
                    #if CDS not used - use last NDR only
                    cds_sig = hdul[idx_2].data 
                    cds_noise = hdul[idx_2 +1].data
            else:
                #use last NDR only
                 cds_sig = hdul[idx_2].data 
                 cds_noise = hdul[idx_2 +1].data
                 
            # perform baackground subtraction
            background_1 = cds_sig[10:16,10:502]
            background_2 = cds_sig[48:54,10:502]
            background = np.mean(np.vstack ( (background_1, background_2) ) )
            cds_sig = cds_sig - background
            
            mask = self.calc_signal_mask(cds_sig)
            cds_sig = cds_sig * mask
            cds_noise = cds_noise * mask
            
            # collect signal and noise into array
            signal[...,i] = cds_sig
            noise[...,i] = cds_noise
        
        # extract spectrum (produces 1 D spectrum from 2 D array)
        signal_stack = np.zeros((nExp, hdul[1].shape[1]))
        noise_stack = np.zeros((nExp, hdul[1].shape[1]))
        for i in range(nExp):
            signal_stack[i] = self.extract_spectrum(signal[...,i]) 
            noise_stack[i] = self.extract_spectrum(noise[...,i]) 
        hdul.close()
        return signal_stack, noise_stack

    def extract_spectrum(self, f): # extracts 1 D spectrum from 2 D image
        # this can become a more sophisticated extraction in the future using Hornes' algorithm
        count = f.sum(axis = 0)
        return count
    
    
    def calc_signal_mask(self, inImage):
        ## Requires Background Subtracted images (?) AP
        # apply mask    
        # 1) find max position of image
        indices = np.where(inImage == inImage.max())
        y_max = indices[0].item()
        x_max = indices[1].item()
        # 2) find FWHM of airy signal at maximum in y direction
        fact = 100
        slice_y = inImage[:, x_max]
        slice_max = np.argmax(slice_y)
        x1 = np.linspace(-1,1,inImage.shape[0])
        x2 = np.linspace(-1,1,inImage.shape[0]+ (inImage.shape[0]-1)*fact ) 
        slice_y2 = np.interp(x2,x1,slice_y)  
        idx01 = np.argwhere(slice_y2 >=  slice_y.max()/2.)[0]
        idx02 = np.argwhere(slice_y2 >=  slice_y.max()/2.)[-1]
        FWHM = (idx02 - idx01) * 1.0 / (fact + 1)
        # 3) find positions of first minima = mask edge
        idx1 = slice_max - np.int(np.ceil(FWHM/ 0.84).item())
        idx2 = slice_max + np.int(np.ceil(FWHM/ 0.84).item())
        # 4) extend mask edges to compensate for likely widening of psf on right side of array
        idx1 = idx1  -3
        idx2 = idx2  +3
        # create and apply mask
        mask =  np.zeros(inImage.shape)
        mask[idx1:idx2+1] = 1.
        return mask
    
    def div_in2_wav_2(self, spec_stack, wl0, R, ld, pix_size): 
            # divides input 1 D array into bins based on R parameter
            x_pix = spec_stack.shape[1]
            m = ld[1]
            bin_size_0 = wl0 / (R* pix_size *m)   # initial bin size in pixel units
            bin_size =[] # bin size in pixel units list
            bin_pos = [] # bin edges position in pixel units list
            tot_pix = 0        
            # bin sizes calculated across array per R starting with bin_size_0
            i = 0
            while tot_pix < x_pix :
                if len(bin_size) == 0:
                    b = (bin_size_0)
                    i += 1
                else:
                    b = ((1+(1/R))* bin_size[i-1])
                    i += 1
                bin_size.append(b)
                tot_pix += b
            bin_size = np.array(bin_size) 
            bin_size_r = np.round(bin_size)        
            bin_pos0 = np.cumsum(bin_size_r) - np.round(bin_size_r[0]/2)
            bin_pos =  np.cumsum(bin_size_r) - np.round(bin_size_r[0]/2)  
            bin_size0 =[]
            for i in range (len(bin_pos)-1):
                bin_size0.append( bin_pos[i+1] - bin_pos[i] )
                
            x_pos = bin_pos*pix_size 
            # find dist of centre of each bin to assign wl
            ##x_cen = []
            x_cen =  np.zeros(len(x_pos)-1)
            for i in range (len(x_cen)):
                x_cen[i] =  (x_pos[i].item() + x_pos[i+1].item() )/ 2
            # find wl at centre of each bin
            wl = ld[0] + ld[1]*(x_cen-ld[2])      
            bin_pos = bin_pos[0:-1]
            bin_pos = bin_pos.tolist()
            # apply binning
            spec = np.add.reduceat(spec_stack, bin_pos, axis = 1)
            return spec, wl.magnitude, bin_pos0
        