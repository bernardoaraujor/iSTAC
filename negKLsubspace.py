
import numpy as np
import scipy
import logdet as ld
#import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def negKLsubspace(k, mu, A, bv, va, vav, vecs, vecssize = 0):

    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

    Local Variables: A, vA, vAb, k, L, mu, v1, bv, vecs, vAv, b1
    Function calls: trace, isempty, norm, negKLsubspace, logdet
    [L] = negKLsubspace(k, mu, A, bv, vA,vAv); 
     
      loss function for computing a subapce which achieves maximal KL-divergence between a
      Gaussian N(mu,A) and N(0,I).  
    
      Computes KL divergence within subspace spanned by [k vecs]
    
      inputs:  
         k = new dimension 
         mu = mean of Gaussian 
         A = cov of Gaussian
    
            Quicker to pass these in than recompute every time:
         bV = mu' * vecs
         vA = vecs'*A
         vAv = vecs'*A*vecs  
    
         vecs = basis for dimensions of subspace already "peeled off"  
    """
    
    if vecssize > 0:
        k = k - np.dot(vecs[:, np.arange(0, vecssize)], (np.dot(vecs[:, np.arange(0, vecssize)].T, k.T)))   # orthogonalize k with respect to 'vecs'
    
    
    k = np.divide(k, np.linalg.norm(k))   # normalize k
    
    b1 = np.dot(k.T, mu)
    v1 = np.reshape(np.dot(k.T, np.dot(A, k)), (1, 1))
    
    if bv.size > 0:
        b1 = np.vstack((np.hstack((b1)), bv))
        vab = np.reshape(np.dot(va, k), (va.shape[0], 1))
        v1 = np.vstack((np.hstack((np.reshape(np.dot(k.T, np.dot(A, k)), (1, 1)), vab.T)), np.hstack((vab, vav))))
    
    
    L = ld.logdet(v1) - np.trace(v1) - np.dot(b1.T, b1)
    
    return L
