import numpy as np

def orthogonalsubset(B, vecs):

    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html

    Local Variables: B, k, j, vorth, etol, vecs, nv
     Function calls: orthogonalsubset, isempty, norm, size

     orthogonalize set of vectors with respect to columns of B ;
     remove those in the column space of B
    """

    etol = 1e-10
    if B.size == 0:
        return vecs

    vorth = np.reshape(np.zeros((vecs.shape[0], 1)), (vecs.shape[0], 1))
    nv = 0
    for j in np.arange(0, vecs.shape[1]):
        k = np.vstack(vecs[:,j] - np.dot(B, np.dot(np.linalg.pinv(B), vecs[:, j])))
        if np.linalg.norm(k) > etol:
            nv = nv+1
            if (nv == 1):
                vorth = np.reshape(k/np.linalg.norm(k), (len(k), 1))
            else:
                vorth = np.append(vorth[:, np.arange(0, vorth.shape[1])], np.reshape(k/np.linalg.norm(k), (len(k), 1)), 1)

    return vorth
