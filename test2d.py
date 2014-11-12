
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

slen = 10000 # Stimulus length
Stim = plt.randn(slen, 1.)
RefreshRate = 100. # refresh rate

DC = (0.75, .5)
linresp = np.hstack((sg.convolve2d(np.concatenate((np.zeros((len(filt1)-1, Stim.shape[1])), Stim), 0), np.rot90(filt1, 2), 'valid')+.75, sg.convolve2d(np.concatenate((np.zeros((len(filt2)-1, Stim.shape[1])), Stim), 0), np.rot90(filt2, 2), 'valid')+.5)) # filter output
r = 10*linresp[:,0]**2 + 8*linresp[:,1]**2      # instantaneous spike rate
r = np.vstack(r)
spikes = np.random.poisson(r/RefreshRate)   # generate spikes
[sta, stc, rawmu, rawcov] = sstc.simpleSTC(Stim, spikes, nt) #% Compute STA and STC
u, s, v = np.linalg.svd(stc, 'full_matrices') # Compute eigenvectors of STC matrix
s = np.diag(s)
v = v.T

ndims = 10
eigvalthresh = 0.05
vecs, vals, mu1, mu0, v1, v2, _ = ci.compiSTAC(sta, stc, rawmu, rawcov, ndims, eigvalthresh)

KLcontributed = np.vstack((np.reshape(vals[0], (1, 1)), np.vstack(np.diff(np.hstack(vals)))))
ndims = len(vals)

fig1 = plt.figure()
plt.plot(np.arange(1, ndims+1), KLcontributed, 'o--')
plt.title('KL contribution')
plt.xlabel('subspace dimensionality')
plt.savefig('plots/KLcontribution')

fig2 = plt.figure()
plt.plot(tvec, np.hstack((filt1, np.dot(u[:,(0, 1)], np.dot(u[:,(0, 1)].T, filt1)), np.dot(vecs[:,(0, 1)], np.dot(vecs[:,(0, 1)].T, filt1)))))
plt.title('Reconstruction of 1st filter')
plt.savefig('plots/reconst_1st_filt.png')

fig3 = plt.figure()
plt.plot(tvec, np.hstack((filt2, np.dot(u[:,(0, 1)], np.dot(u[:,(0, 1)].T, filt2)), np.dot(vecs[:,(0, 1)], np.dot(vecs[:,(0, 1)].T, filt2)))))
plt.title('Reconstruction of 2nd filter')
plt.savefig('plots/reconst_2nd_filt.png')

fig4 = plt.figure()
plt.plot(tvec, np.hstack((np.vstack(vecs[:, 0]), np.vstack(vecs[:, 1]))))
plt.title('iSTAC filters')
plt.savefig('plots/iSTAC_filters')

a = 2
