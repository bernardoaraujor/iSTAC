
import numpy as np
import scipy
import logdet as ld
# import matcompat
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass


def compDklgaussian(mu1, C1, mu2, C2):
    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

    Local Variables: b, mu1, d, Term1, mu2, DD, Term2, Term3, n, C2, C1, C1divC2
    Function calls: compDklgaussian, nargin, length, trace, logdet
    d = compDklgaussian(mu1, C1, mu2, C2)
    
    Computes the KL divergence between two multivariate Gaussians
    
    Inputs:
     mu1 = mean of 1st Gaussian
     C1 = covariance of 1st Gaussian
     mu2 = mean of 2nd Gaussian
     C2 = covariance of 2nd Gaussian
    
    Notes:
      D_KL = Integral(p1 log (p1/p2))
    Analytically:  (where |*| = Determinant, Tr = trace, n = dimensionality
         =  1/2 log(|C2|/|C1|) + 1/2 Tr(C1^.5*C2^(-1)*C1^.5)
            + 1/2 (mu2-mu1)^T*C2^(-1)*(mu2-mu1) - 1/2 n
"""
    n = len(mu1)
    b = mu2 - mu1
    C1divC2 = np.linalg.solve(C2, C1) # matrix we need    
    
    Term1 = np.trace(C1divC2) # trace term    
    Term2 = np.dot(b.conj().T, np.linalg.solve(C2, b)) # quadratic term    
    #TEST THIS!!!
    Term3 = np.negative(ld.logdet(C1divC2))
    
    d = np.dot(.5, Term1 + Term2 + Term3 - n)
    return d
