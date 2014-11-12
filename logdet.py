
import numpy as np
import scipy
#import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def logdet(A):
    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

    Local Variables: A, x
    Function calls: diag, sum, chol, log, logdet
     LOGDET - computes the log-determinant of a matrix A
    
     x = logdet(A);
    
     This is faster and more stable than using log(det(A))
    
     Input:
         A NxN - A must be sqaure, positive semi-definite
    """
    
    x = 2.*np.sum(np.log(np.diag(np.linalg.cholesky(A))))
    return x