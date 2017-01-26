iSTAC
=====
info-theoretic spike-triggered average and covariance (iSTAC) - python code

estimates a set of linear filters (or "receptive fields") that best capture a neuron's input-output properties, 
using an information-theoretic objective that optimally combines spike-triggered average and spike-triggered, 
covariance information. The filters can be considered as the first stage in a linear-nonlinear-Poisson (LNP), 
model of the neuron's response. They are sorted by informativeness, providing an estimate of the mutual information, 
gained by the inclusion of each filter. 

Adapted by Bernardo Rodrigues from J. Pillow's MATLAB code: https://github.com/pillowlab/iSTAC

===========================================
- compiSTAC.py: Computes a set of iSTAC "filters" -- i.e., an orthogonal basis that captures the maximal information about spiking given the stimuli.  This is equal to the basis that captures maximal KL divergence between the two Gaussians N(mu1,A1) and N(mu0,A0).

- simpleSTC.py: Computes mean and covariance of raw and spike-triggered stimuli.

all other functions are called by the two above.

Using:

vecs, vals, mu1, mu0, v1, v2, nullBasis = compiSTAC(mu1, A1, mu0, A0, ndims, cthr=1e-2)

Inputs:
-                    mu1 [n x 1] = spike-triggered average    (1/nsp*sum_i y_i*x_i)
-                    A1 [n x n] = spike-triggered covariance (with mean removed)
-                    mu2 [n x 1] = mean of stimulus ensemble (1/N sum_i x_i)
-                    A2 [n x n] = cov of stim ensemble (1/N sum_i x_i x_i - mu2*mu2^T
-                    ndims [1 x 1] = number of filters to estimate 
-                    cthr  [1 x 1] = eigenvalue threshold for whitening (OPTIONAL; DEFAULT=0.01). Will project out any dimensions for which the variance of the raw stimuli is < max(eigval)*cthr. 

Outputs: 
-                    vecs [n x ndims] = matrix with columns giving an (ordered) basis for the maximal information-preserving subspace of degree ndims
-                    vals [ndims x 1] = value of the KL divergence as subspace dimensionality increases from 1 to ndims 
-                    mu1 = means of first gaussian
-                    A1 = covariance matrix of first gaussian
-                    mu2 = means of second gaussian
-                   A2 = covariance matrix of second gaussian
-                    nullBasis [n x m] = matrix with columns giving basis for undersample subspace of raw stimuli (i.e., which was ignored)   

-----------------------------------------------------------

STA, STC, RawMu, RawCov = simpleSTC(Stim, sp, nkt)

Inputs:
-                    Stim = stimulus matrix; 1st dimension = time, 2nd dimension = space
-                    sp = spikes;  coloumn vector of spike count in each time bin or list of spike times
-                    nkt  = number of time samples to include in the spike-triggered ensemble
-                    CriticalSize = maximum number of floats to store while computing cov (smaller = slower but smaller memory requirement)   
                     
Outputs:
-                    STA = spike-triggered average
-                    STC = spike-triggered covariance (covariance around the mean);
-                    RawMu = mean of raw stimulus ensemble
-                    RawCov = covariance of raw ensemble
