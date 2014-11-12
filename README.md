iSTAC
=====
info-theoretic spike-triggered average and covariance (iSTAC) - python code

estimates a set of linear filters (or "receptive fields") that best capture a neuron's input-output properties, 
using an information-theoretic objective that optimally combines spike-triggered average and spike-triggered, 
covariance information. The filters can be considered as the first stage in a linear-nonlinear-Poisson (LNP), 
model of the neuron's response. They are sorted by informativeness, providing an estimate of the mutual information, 
gained by the inclusion of each filter. 

Adapted from MATLAB code by J. Pillow: http://pillowlab.cps.utexas.edu/code_iSTAC.html

Report any bugs to bernardoaraujor@gmail.com
