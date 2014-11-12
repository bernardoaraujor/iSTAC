
import numpy as np
import scipy as sp
#import matcompat

# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def gsorth(a, B = None):
    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

    Local Variables: a, B, m, j, v
    Function calls: gs, nargin, orth, gsorth, norm, size
      Performs Graham-Schmidt orthogonalization
    
      V = gsorth(B),  or V = gramschm(a, B)
        if B is a matrix, then V will be an orthonormal matrix whose 
        column vectors have been formed by successive GS orthogonlization
    
      V = gramschm(a, B)
        two args:  a is a vector, B a matrix.  Returns unit vector 
        which has been orthogonalized to B.
    """

    if (a.shape[1] > 1) & (B is not None):
        print('gsorth: Wrong inputs')
        return
    elif (a.shape[1] >= 1) & (B is None):
         m = a.shape[1]
         v = np.zeros((a.shape[0], a.shape[1]))
         v[:, 0] = np.reshape(np.vstack(np.divide(a[:,0], np.linalg.norm(a[:,0]))), (a.shape[0]))
         for j in np.arange(1, m):
            v[:,j] = np.reshape(gs(a[:,j], v), (a.shape[0]))

    return v
    


def gs(v, B):
    """
    Local Variables: vnew, B, v
    Function calls: fprintf, gs, norm
    
    Orthogonalizes v wrt B;  assumes that B is orthogonal
    """
    v = np.vstack(np.divide(v, np.linalg.norm(v)))
    vnew = v-np.dot(B, np.dot(B.T, v))
    if np.linalg.norm(vnew) > 1e-10 :
        vnew = np.divide(vnew, np.linalg.norm(vnew))
    else :
        print('\nERROR (gsorth):  vector is linearly dependent\n')
        print('Returning non-unit vector\n')
    
    return vnew