
import numpy as np
import scipy
import scipy.linalg
#import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def makeStimRows(Stim, n, flag = 0):
    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

    Local Variables: sz, Stim, nsp1, j, flag, n, S, n2
    Function calls: toeplitz, max, makeStimRows, fliplr, reshape, min, nargin, length, isempty, error, zeros, prod, size
      S = makeStimRows(Stim, n, flag);
    
      Converts raw stimulus to a matrix where each row is loaded with the full
      space-time stimulus at a particular moment in time.  The resulting
      matrix has length equal to the number of time points and width equal to
      the (number of spatial dimensions) x (kernel size n).
    
      Inputs: 
       Stim = stimulus, first dimension is time, other dimensions are spatial
              dimensions
       n = size of temporal kernel; number of time samples to include in each
           row of stimulus matrix.
       flag (optional)
            = 0, default behavior: padded w/ zeros at beginning so length of
              output matrix matches size of Stim
            = 1, no padding with zeros: length of S is length(Stim)-n+1.
            = vector of indices, (e.g. indices of spikes times).  Return
            a matrix containting only the spiking stimuli
    
      Output: S = matrix where each row is the size of the linear kernel
    """
        
    sz = Stim.shape
    n2 = np.prod(sz[1:])   # total dimensionality in spatial dimensions
    
    # If necessary, convert Stim to a 2D matrix
    if n2 > sz[1]:   # reshape to matrix if necessary
        Stim = np.reshape(Stim, sz[0], n2)
    
    if flag == 0.:   # Compute with zero-padding. ----------------------------------
        S = np.zeros((sz[0], np.dot(n2, n)))
        for j in np.arange(1, n2+1):
            S[:,np.dot(n, j-1):np.dot(n, j)-1] = np.fliplr(scipy.linalg.toeplitz(Stim[:,j-1], np.array(np.hstack((Stim[0,j-1], np.zeros((1., n-1.)))))))
        
    elif flag == 1.:   # compute only for stimuli at times >= n ----------
        S = np.empty((sz[0]-n+1., np.dot(n2, n)))
        for j in np.arange(1, n2+1):
            S[:,np.arange(np.dot(n, j-1), np.dot(n, j))] = np.fliplr(scipy.linalg.toeplitz(Stim[np.arange(n-1, sz[0]),j-1], Stim[np.arange(n-1, -1, -1), j-1]))

            
    else:  # compute for spiking stimuli --------------------------------------------
        if np.logical_or(min(flag)<1., max(flag) > sz[0]):
            raise Exception("makeStimRows:  3rd arg should be spike indices (vals are too high or too low): %d %d" % (min(flag), max(flag)))
        
        S = np.zeros(len(flag), np.dot(n2, n))
        
        # Do for spinds < n
        nsp1 = len(flag[flag<n])
        for j in np.arange(1., nsp1+1):
            S[j-1,:] = np.reshape(np.array(np.vstack(np.zeros((n-flag[j-1], n2), Stim[0:flag[j-1],:]))), 1., np.dot(n2, n))
            
        for j in np.arange(nsp1+1., len(flag)+1):
            S[j-1,:] = np.reshape(Stim[flag[j-1]-n:flag[j-1],:], 1., np.dot(n2, n))        
         
    return S