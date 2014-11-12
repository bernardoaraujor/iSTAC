
import numpy as np
import scipy as sp
import scipy.signal as sg
import simpleSTC as sstc
import compiSTAC as ci
import matplotlib

try:
    import matplotlib.pylab as plt
except ImportError:
    pass

nt = 20 # number of temporal elements of filter
tvec = np.vstack(np.arange(-nt+1., 1)) # time vector
filt1 = np.exp(-(((tvec+4.5)/1.5)**2)/2) - .2*np.exp((-((tvec+nt/2)/3)**2)/2)  #1st filter
filt1 = filt1/np.linalg.norm(filt1) #normalize
filt2 = np.vstack((np.diff(filt1, 1, 0), 0)) # 2nd filter
filt2 = filt2 - filt1*np.sum(filt1*filt2) # orthogonalize to 1st filter
filt2 = (filt2)/np.linalg.norm(filt2) # normalize

slen = 2000 # Stimulus length
Stim = plt.randn(slen, 1.)
RefreshRate = 100. # refresh rate

linresp = (sg.convolve2d(np.concatenate((np.zeros((len(filt1)-1, Stim.shape[1])), Stim), 0), np.rot90(filt1, 2), 'valid')) # filter output
r = np.maximum(linresp, 0.)*50      # instantaneous spike rate
r = np.vstack(r)
spikes = np.random.poisson(r/RefreshRate)   # generate spikes
[sta, stc, rawmu, rawcov] = sstc.simpleSTC(Stim, spikes, nt) #% Compute STA and STC
u, s, v = np.linalg.svd(stc, 'full_matrices') # Compute eigenvectors of STC matrix
s = np.diag(s)
v = v.T

ndims = 1
vecs, vals, mu1, mu0, v1, v2, _ = ci.compiSTAC(sta, stc, rawmu, rawcov, ndims)

fig1 = plt.figure()
plt.plot(tvec, np.hstack((filt1, sta/np.linalg.norm(sta), np.reshape(u[:, -1], (len(u), 1)), vecs)))
plt.title('Test: rectified linear LNP neuron')
plt.savefig('plots/test1d')

a = 2
