
import numpy as np
import numpy.matlib as ml
import scipy
import scipy.io
import makeStimRows as msr
#import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def simpleSTC(Stim, sp, nkt, CriticalSize = 1e8):
    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

      [STA,STC,RawMu,RawCov] = simpleSTC(Stim, sp, nkt);
    
      Computes mean and covariance of raw and spike-triggered stimuli
    
      Input:  
       Stim = stimulus matrix; 1st dimension = time, 2nd dimension = space
       sp = spikes;  coloumn vector of spike count in each time bin
                     or list of spike times
       nkt  = number of time samples to include in the spike-triggered ensemble
       CriticalSize - maximum number of floats to store while computing cov
                      (smaller = slower but smaller memory requirement)
    
      Output:
       STA = spike-triggered average
       STC = spike-triggered covariance (covariance around the mean);
       RawMu = mean of raw stimulus ensemble
       RawCov = covariance of raw ensemble
    
      Notes:  
        (1) raw spike-triggered 2nd moment = STC+STA(:)*STA(:)'*nsp/(nsp-1)
        (2) condition (Msz >= CriticalSize) needs to be fixed

     Translated from MATLAB into Python by Bernardo Rodrigues.
     Copyright 2010 Pillow Lab. All rights reserved.
     """
    
    [slen, swid] = Stim.shape  # stimulus size (time bins x spatial bins).
    
    if len(sp) != Stim.shape[0]:
        # Convert list of spike times to spike-count vector
        print('simpleSTC: converting spike times to counts\n')
        sp = plt.hist(sp, np.array(np.hstack((np.arange(1., (slen)+1))))).T
    
    sp[0:nkt-2] = 0.  # Ignore spikes before time n
    
    # ---------------------------------------------------
    # Compute both the spike-triggered and raw stimulus means and covariances
    
    sp = sp[nkt-1:]  # Take spikes only from the nth index onward
    slen = len(sp)  # length of time indices for stim and spikes
    swid = Stim.shape[1]
    nsp = np.sum(sp)  # number of spikes
    Msz = np.dot(np.dot(slen, swid), nkt)  # Size of full stimulus matrix
    rowlen = np.dot(swid, nkt) # Length of a single row of stimulus matrix
    
    if Msz<CriticalSize:  # Check if stimulus is small enough to do in one chunk
        SS = msr.makeStimRows(Stim, nkt, 1.)  # Convert stimulus to matrix where each row is one stim
        
        # Compute raw mean and covariance
        RawMu = np.mean(SS, 0).T
        RawCov = np.dot(SS.T, SS) / (slen-1.) - (RawMu*np.vstack(RawMu)*slen) / (slen-1.)
        
        # Compute spike-triggered mean and covariance
        iisp = np.nonzero((sp > 0.))
        spvec = sp[iisp]
        STA = np.divide(np.dot(spvec.T, SS[iisp[0],:]).T, nsp)
        STC = np.dot(SS[iisp[0],:].T, np.multiply(SS[iisp[0],:], ml.repmat(spvec, rowlen, 1).T))/(nsp-1.) - (STA*np.vstack(STA)*nsp)/(nsp-1.)
    
    #-----------------------------------------------------------
    # NEEDS TO BE FIXED!
    #-----------------------------------------------------------
    else:   # Compute Full Stim matrix in chunks, compute mean and cov on chunks
        nchunk = np.ceil(np.divide(Msz, CriticalSize))
        chunksize = np.ceil(np.divide(slen, nchunk))
        print('simpleSTC: using %d chunks to compute covariance\n' % (nchunk))
        
        # Compute statistics on first chunk
        SS = msr.makeStimRows(Stim[0:chunksize+nkt-2.,:], nkt, 1.)  # convert stimulus to "full" version
        spvec = sp[0:chunksize-1]
        iisp = np.nonzero((spvec > 0.))
        RawMu = np.sum(SS).T
        RawCov = np.dot(SS.T, SS)
        STA = np.dot(spvec[iisp].T, SS[iisp,:]).T
        STC = np.dot(SS[iisp,:].T, np.dot(SS[iisp,:], np.tile(spvec[iisp], (1., rowlen))))
        
        # add to mean and covariance for remaining chunks
        for j in np.arange(2., (nchunk)+1):
            i0 = np.dot(chunksize, j-1.)+1  # starting index for chunk
            imax = min(slen, np.dot(chunksize, j))  # ending index for chunk
            SS = msr.makeStimRows(Stim[i0-1:imax+nkt-2,:], nkt, 1.)
            spvec = sp[i0-1:imax]
            iisp = np.nonzero(spvec)
            
            RawMu = RawMu+np.sum(SS).T
            RawCov = RawCov+np.dot(SS.T, SS)
            STA = STA+np.dot(spvec[iisp].T, SS[iisp:]).T
            STC = STC+np.dot(np.dot(SS[iisp-1,:].T, SS[iisp-1,:], np.tile(spvec[int(iisp)-1], (1., rowlen))))
                
            # divide means and covariances by number of samples
            RawMu = np.divide(RawMu, slen)
            RawCov = np.divide(RawCov, slen-1.) - np.divide(np.dot(RawMu, np.dot(RawMu.T, slen)), slen-1.)
            
            STA = np.divide(STA, nsp)
            STC = np.divide(STC, nsp-1.)-np.divide(np.dot(STA, np.dot(STA.T, nsp)), nsp-1.)   
     
    #KEEP THIS?    
    #STA = np.reshape(STA, np.array([]), swid)
    
    return [np.vstack(STA), STC, np.vstack(RawMu), RawCov]
