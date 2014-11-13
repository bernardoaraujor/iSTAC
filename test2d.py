
import numpy as np
import scipy.signal as sg
import simpleSTC as sstc
import compiSTAC as ci
import matplotlib.pylab as plt

nt = 20 # number of temporal elements of filter
tvec = np.vstack(np.arange(-nt+1., 1)) # time vector
filt1 = np.exp(-(((tvec+4.5)/1.5)**2)/2) - .2*np.exp((-((tvec+nt/2)/3)**2)/2)  #1st filter
filt1 = filt1/np.linalg.norm(filt1) #normalize
filt2 = np.vstack((np.diff(filt1, 1, 0), 0)) # 2nd filter
filt2 = filt2 - filt1*np.sum(filt1*filt2) # orthogonalize to 1st filter
filt2 = (filt2)/np.linalg.norm(filt2) # normalize

x = np.arange(0, 20)
filt3 = (x-10)**2
filt3[0:10] = 0
filt3 = (filt3)/np.linalg.norm(filt3)

filt1 = np.reshape(filt1, (nt, 1))
filt2 = np.reshape(filt2, (nt, 1))
filt3 = np.reshape(filt3, (nt, 1))

slen = 100000 # Stimulus length
#mat = sp.io.loadmat('C:/Users/bernardo/Documents/MATLAB/USP/iSTAC/stimtest.mat')
#Stim = mat['Stim']
Stim = plt.randn(slen, 1.)
RefreshRate = 100. # refresh rate

DC = (0.75, .5)
linresp = np.hstack((sg.convolve2d(np.concatenate((np.zeros((len(filt1)-1, Stim.shape[1])), Stim), 0), np.rot90(filt1, 2), 'valid')+.75, sg.convolve2d(np.concatenate((np.zeros((len(filt2)-1, Stim.shape[1])), Stim), 0), np.rot90(filt2, 2), 'valid')+.5, sg.convolve2d(np.concatenate((np.zeros((len(filt3)-1, Stim.shape[1])), Stim), 0), np.rot90(filt3, 2), 'valid')+.5)) # filter output
r = 10*linresp[:,0]**2 + 10*linresp[:,1]**2 + 5*linresp[:,2]**2     # instantaneous spike rate
r = np.vstack(r)
#spikes = mat['spikes']
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
plt.show()
#plt.savefig('plots/KLcontribution')

fig2 = plt.figure()
f1, = plt.plot(tvec, filt1)
stc1, = plt.plot(tvec, np.dot(u[:,(0, 1)], np.dot(u[:,(0, 1)].T, filt1)))
istac1, = plt.plot(tvec, vecs[:, 0])
plt.legend([f1, stc1, istac1], ['true k', 'STC', 'iSTAC'], loc = 2)
plt.title('1st Kernel')
plt.show()
#plt.savefig('plots/reconst_1st_filt.png')

fig3 = plt.figure()
f2, = plt.plot(tvec, filt2)
stc2, = plt.plot(tvec, np.dot(u[:,(0, 1)], np.dot(u[:,(0, 1)].T, filt2)))
istac2, = plt.plot(tvec, vecs[:, 1])
plt.legend([f2, stc2, istac2], ['true k', 'STC', 'iSTAC'], loc = 2)
plt.title('2nd Kernel')
plt.show()
#plt.savefig('plots/reconst_2nd_filt.png')

fig32 = plt.figure()
f3, = plt.plot(tvec, filt3)
stc2, = plt.plot(tvec, np.dot(u[:,(0, 1)], np.dot(u[:,(0, 1)].T, filt2)))
istac2, = plt.plot(tvec, vecs[:, 2])
plt.legend([f2, stc2, istac2], ['true k', 'STC', 'iSTAC'], loc = 2)
plt.title('3rd Kernel')
plt.show()
#plt.savefig('plots/reconst_2nd_filt.png')

