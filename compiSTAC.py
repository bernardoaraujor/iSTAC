
import numpy as np
import scipy as sp
import negKLsubspace
import compDklgaussian as cdkl
import orthogonalsubset as os
import gsorth as gs
import negKLsubspace as nkls
#import matcompat
# if available import pylab (from matlibplot)
try:
    import matplotlib.pylab as plt
except ImportError:
    pass

def compiSTAC(mu1, A1, mu0, A0, ndims, cthr=1e-2):
    """
    Adapted from:
    http://pillowlab.cps.utexas.edu/code_iSTAC.html
    Translated from MATLAB into Python by Bernardo Rodrigues.

    Local Variables: GaussParams, nullBasis, ii, valmarginals, valtst, vv, valdiffs, uvecs, vA, k0s, LB, v0s, imin, nd, A1, A0, k0, A, cthr, kstrt, BackingUP, bv, sdiag, vecs, jj, vals, Beq, vAv, a, mu1, mu0, v1, k, j, v2, iikeep, mu, ndims, Wmat, opts, UB
    Function calls: fprintf, diff, find, size, orthogonalsubset, eye, min, diag, sum, sqrt, zeros, norm, exitflag, nargin, ones, compDklgaussian, gsorth, svd, floor, length, u, compiSTAC, optimset, k, negKLsubspace
     [vecs, vals, GaussParams,nullBasis] = compiSTAC(mu1,A1,mu0,A0,ndims,cthr)
    
     Computes a set of iSTAC filters -- i.e., an orthogonal basis that
     captures the maximal information about spiking given the stimuli.
     This is equal to the basis that captures maximal KL divergence between
     the two Gaussians N(mu1,A1) and N(mu0,A0).
    
     Implicitly, N(mu1,A1) is a Gaussian description of the spike-triggered
     ensemble, while N(mu0,A0) is a description of the raw stim distribution
    
     Whitens with respect to N(mu0,A0), so computation is simplified.
    
     Inputs:  
       mu1 [n x 1] = spike-triggered average    (1/nsp*sum_i y_i*x_i)
        A1 [n x n] = spike-triggered covariance (with mean removed)
                              (1/nsp*sum_i y_i*x_i*x_i^T - mu1*mu1^T)
       mu2 [n x 1] = mean of stimulus ensemble (1/N sum_i x_i)
        A2 [n x n] = cov of stim ensemble (1/N sum_i x_i x_i - mu2*mu2^T
     ndims [1 x 1] = number of filters to estimate 
     cthr  [1 x 1] = eigenvalue threshold for whitening (OPTIONAL; DEFAULT=0.01).
                     Will project out any dimensions for which the variance of
                     the raw stimuli is < max(eigval)*cthr. 
    
     Ouptuts: 
       vecs [n x ndims] = matrix with columns giving an (ordered) basis for the 
                          maximal information-preserving subspace of degree ndims
       vals [ndims x 1] = value of the KL divergence as subspace dimensionality increases
                          from 1 to ndims 
       GaussParams - structure containing the means and covariances of the two
                     Gaussians projected into the subspace of interest.
                    (Useful if we wish to use ratio-of-Gaussians to describe the
                     nonlinearity).
       nullBasis [n x m] = matrix with columns giving basis for undersampled
                           subspace of raw stimuli (i.e., which was ignored)   
    """
    
    
    #Initialize some optimization params
    vecs = np.array([])
    
    #USE THIS?
    #opts = optimset('display', 'off', 'gradobj', 'off', 'largescale', 'off', 'maxfunevals', 200000., 'maxiter', 50., 'algorithm', 'Active-set')
    
    #Compute whitening matrix
    uvecs, sdiag, _ = np.linalg.svd(A0)  # eigenvalues of raw stimulus covariance
    sdiag = np.vstack(sdiag)

    if sdiag[-1]/sdiag[0] > cthr:    # check condition number
        # Keep full space       
        Wmat = np.dot(np.diag(np.hstack(1/np.sqrt(sdiag))), uvecs.T)   # whitening matrix
        nullBasis = np.array([])
    else:
        # prune some dimensions

        #NOT YET DEBUGGED!
        #CHANGE THIS?
        iikeep = sdiag > np.dot(sdiag[0], cthr)        
        Wmat = np.dot(np.diag(np.hstack(np.divide(1., np.vstack(np.sqrt(sdiag[iikeep]))))), uvecs[:, np.hstack(iikeep)].T)
        nullBasis = uvecs[:, np.hstack(np.invert(iikeep))]
        print('Pruning out %d dimensions (out of %d) from raw stimuli\n' % (np.count_nonzero(np.invert(iikeep)), len(sdiag)))
            
    mu = np.dot(Wmat, mu1-mu0)
    A = np.dot(Wmat,np.dot(A1, Wmat.T))
    
    # Set upper and lower bounds for optimization
    nd = len(mu)
    UB = np.ones((nd, 1))
    LB = -np.ones((nd, 1))
                  
    # Compute SVD of whitened covariance, for initial guesses
    u, _, _ = np.linalg.svd(A)
    a = int(min(ndims, np.floor(nd/2)))
    k0s = np.hstack((u[:, np.hstack((np.arange(0, a), np.arange(len(u)-a, len(u))))], mu/np.linalg.norm(mu)))

    bv = np.array([])
    vA = np.array([])
    vAv = np.array([])
    j = 0

    flag = 0
    while j <= min(ndims, (nd-1))-1:
        BackingUP = 0
        
        # Start by finding best starting point for optimization
        if j == 4:
            o = 2
        kstrt = os.orthogonalsubset(vecs, k0s)
        v0s = np.zeros((kstrt.shape[1], 1))
        for ii in np.arange(0, kstrt.shape[1]):
            if j == 0:
                v0s[ii] = nkls.negKLsubspace(np.vstack(kstrt[:,ii]), mu, A, bv, vA, vAv, vecs)
            else:
                if (ii == 12) and (j == 3):
                    o = 2
                v0s[ii] = nkls.negKLsubspace(np.vstack(kstrt[:,ii]), mu, A, bv, vA, vAv, vecs[:, np.arange(0, j)])
        v0s = v0s.T

        imin = np.nonzero(np.array((1*(v0s == np.amin(v0s)))))[1]
        k0 = kstrt[:,imin]
        
        # Perform optimization -- restart if optimization doesn't terminate
        Beq = np.zeros((j, 1))
        
        #IMPLEMENT THIS?
        #[k,~,exitflag] = fmincon(@negKLsubspace, k0,[],[],vecs',Beq,LB,UB,...
         #       @NormEq1,opts,mu,A,bv,vA,vAv,vecs);

        opt = ({'maxiter': '50', 'disp': False})
        if len(vecs) > 0:
            cons = ({'type': 'eq', 'fun': np.dot(vecs[:, np.arange(0, len(Beq))].T, k0) - Beq}, {'type': 'ineq', 'fun': np.dot(k0.T, k0)-1}, {'type': 'eq', 'fun': 2*k0})
            #k = sp.optimize.minimize(nkls.negKLsubspace, k0, args = (mu, A, bv, vA, vAv, vecs), bounds = np.hstack((LB, UB)), constraints = cons, options = opt)
            k = sp.optimize.minimize(nkls.negKLsubspace, k0, args = (mu, A, bv, vA, vAv, vecs), bounds = np.hstack((LB, UB)))['x']
            k = np.vstack(k)

        else:
            k = sp.optimize.minimize(nkls.negKLsubspace, k0, args = (mu, A, bv, vA, vAv, vecs), bounds = np.hstack((LB, UB)))['x']
            k = np.vstack(k)

        #if exitflag<1:
            #% Check convergence
            #print('iSTAC-- possible error: optimization not terminated; j=%d\nFIX THIS?\n'%j)
        
        if (j > 0):    # normalize k with respect to previous vecs
            k = k - np.dot(vecs, np.dot(vecs.T, k))
            k = k / np.linalg.norm((k))
        
        # Compute KL divergence along this dimension
        print(j)
        print(k)
        if j == 0:
            vecs = np.zeros((len(k), min(ndims, (nd-1))))
            vals = np.zeros((min(ndims, (nd-1)), 1))
            valmarginals = np.zeros((min(ndims, (nd-1)), 1))
            valdiffs = np.hstack((vals[0]))

        vecs[:,j] = np.reshape(k, (vecs[:, j].shape))
        vals[j,0] = np.reshape(cdkl.compDklgaussian(np.dot(vecs[:, np.arange(0, j+1)].T, mu), np.dot(vecs[:, np.arange(0, j+1)].T, np.dot(A, vecs[:, np.arange(0, j+1)])), np.zeros((j+1, 1.)), np.eye(j+1)), vals[j,0].shape)

        if j != 0:
            valdiffs = np.vstack((np.hstack((vals[0])), np.reshape((np.diff((vals[np.arange(0, j+1)]).T)), (j, 1))))


        valmarginals[j,0] = np.reshape(cdkl.compDklgaussian(np.dot(k.T, mu), np.dot(k.T, np.dot(A, k)), np.zeros((1, 1.)), np.eye(1)), valmarginals[j,0].shape)

        # Check that vals is smaller than all previous values
        if BackingUP >= 3.:
            BackingUP = 0.

        if len(valdiffs) <= 2:
            mini = 0
        else:
            mini = min(valdiffs[0:j])

        if (valdiffs[j] > mini) and (j+1 < nd/2.) and (mini != 0) and (j < nd/2):
            jj = np.nonzero((valdiffs[0:j] < valdiffs[j]))
            k0s = np.array(np.hstack((k, k0s)))

            vecss = vecs[:,0:jj[0]]
            valss = vals[0:jj[0]]
            vecs = np.zeros((len(k), min(ndims, (nd-1))))
            vals = np.zeros((min(ndims, (nd-1)), 1))

            vecs[:,0:jj[0]] = vecss
            vals[0:jj[0]] = valss
            j = jj[0]
            print('Going back to iter #%d (valdiff=%.4f)\n' % (j, valdiffs[-1]))
            BackingUP = 1
            flag = 1
            
        elif j > 0.:
            vv = vecs[:,np.array(np.hstack((np.arange(0, j-1), j)))]
            valtst = cdkl.compDklgaussian(np.dot(vv.T, mu), np.dot(vv.T, np.dot(A,vv)), np.zeros(((j), 1)), np.eye((j)))
            if valtst > vals[j-1] + 1e-4:
                print('Wrong dim possibly stripped off [%.4f %.4f]; going back to prev dim\n' % (vals[j-1], valtst))
                k0s = np.array(np.hstack((k, k0s)))

                vecs = vecs[:,0:j-2]
                vals = vals[0:j-2]
                j = j-1
                BackingUP = BackingUP+1.      
                    
        if not BackingUP:
            print('Stripped off dimension %d, KL div=[%2.4f %2.4f]\n' % (j, valdiffs[j-1], valmarginals[j-1]))
            j = j+1
            bv = np.dot(vecs[:, np.arange(0, j)].T, mu)
            vA = np.dot(vecs[:, np.arange(0, j)].T, A)
            vAv = np.dot(vecs[:, np.arange(0, j)].T, np.dot(A, vecs[:, np.arange(0, j)]))
        else:
            bv = np.dot(vecs[:, np.arange(0, j)].T, mu)
            vA = np.dot(vecs[:, np.arange(0, j)].T, A)
            vAv = np.dot(vecs[:, np.arange(0, j)].T, np.dot(A, vecs[:, np.arange(0, j)]))

    if j == 10:
        o = 2

    vecs = np.dot(Wmat.T, vecs[:, np.arange(0, j)])
    vecs = gs.gsorth(vecs)

    # REMOVE GAUSSPARAMS?
    mu1 = np.dot(vecs.T, mu1)
    mu0 = np.dot(vecs.T, mu0)
    v1 = np.dot(vecs.T, np.dot(A1, vecs))
    v2 = np.dot(vecs.T, np.dot(A0, vecs))
    
    return vecs, vals, mu1, mu0, v1, v2, nullBasis

def NormEq1(x):
    """
    nonlinear function for implementing the constraint norm(x) = 1;
    """

    ceq = np.dot(x.T, x)-1.
    dceq = 2.*x
        
    return [ceq, dceq]

