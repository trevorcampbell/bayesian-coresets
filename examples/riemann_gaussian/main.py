import numpy as np
import bayesiancoresets as bc
import os, sys
from scipy.stats import multivariate_normal
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import gaussian


M = 200
N = 1000
d = 200
opt_itrs = 500
proj_dim = 100
pihat_noise =0.75

mu0 = np.zeros(d)
Sig0 = np.eye(d)
Sig = np.eye(d)
SigL = np.linalg.cholesky(Sig)
th = np.ones(d)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
SigLInv = np.linalg.inv(SigL)

nm = sys.argv[1]
tr = sys.argv[2]

#generate data and compute true posterior
#use the trial # as the seed
np.random.seed(int(tr))

x = np.random.multivariate_normal(th, Sig, N)
mup, Sigp = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
Sigpinv = np.linalg.inv(Sigp)

#for the algorithm, use the trial # and name as seed
np.random.seed(int(''.join([ str(ord(ch)) for ch in nm+tr])) % 2**32)

#compute constants for log likelihood function
xSiginv = x.dot(Siginv)
xSiginvx = (xSiginv*x).sum(axis=1)
logdetSig = np.linalg.slogdet(Sig)[1]

log_likelihood = lambda samples : gaussian.gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples)

#create tangent space for well-tuned Hilbert coreset alg
T_true = bc.MonteCarloFiniteTangentSpace(log_likelihood, lambda sz : np.random.multivariate_normal(mup, Sigp, sz), proj_dim)

#create tangent space for poorly-tuned Hilbert coreset alg
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
T_noisy = bc.MonteCarloFiniteTangentSpace(log_likelihood, lambda sz : np.random.multivariate_normal(muhat, Sighat, sz), proj_dim)

#create exact tangent space factory for Riemann coresets
def tangent_space_factory(wts, idcs):
  w = np.zeros(x.shape[0])
  w[idcs] = wts
  muw, Sigw = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, w)
  nu = (x - muw).dot(SigLInv.T)
  Psi = np.dot(SigLInv, np.dot(Sigw, SigLInv.T))

  nu = np.hstack((nu.dot(np.linalg.cholesky(Psi)), 0.25*np.sqrt(np.trace(np.dot(Psi.T, Psi)))*np.ones(nu.shape[0])[:,np.newaxis]))
  
  return bc.FixedFiniteTangentSpace(nu, wts, idcs)
  
def nulltsf(wts, idcs):
  return bc.FixedFiniteTangentSpace(np.zeros((x.shape[0], 2)), wts, idcs)
 
 
#create coreset construction objects
riemann_one = bc.SparseVICoreset(x.shape[0], tangent_space_factory, opt_itrs=opt_itrs, update_single=True)
riemann_full = bc.SparseVICoreset(x.shape[0], tangent_space_factory, opt_itrs=opt_itrs, update_single=False)
giga_true = bc.GIGACoreset(T_true)
giga_noisy = bc.GIGACoreset(T_noisy)
unif = bc.UniformSamplingKLCoreset(x.shape[0], nulltsf)

algs = {'SVI1': riemann_one, 
        'SVIF': riemann_full, 
        'GIGAT': giga_true, 
        'GIGAN': giga_noisy, 
        'RAND': unif}
alg = algs[nm]

w = np.zeros((M+1, x.shape[0]))
for m in range(1, M+1):
  print('trial: ' + tr +' alg: ' + nm + ' ' + str(m) +'/'+str(M))

  alg.build(m, 1)
  #store weights
  wts, idcs = alg.weights()
  w[m, idcs] = wts

  #printouts for debugging purposes
  #print('reverse KL: ' + str(weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)))
  #print('reverse KL opt: ' + str(weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)))

muw = np.zeros((M+1, mu0.shape[0]))
Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
rklw = np.zeros(M+1)
fklw = np.zeros(M+1)
for m in range(M+1):
  muw[m, :], Sigw[m, :, :] = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, w[m, :])
  rklw[m] = gaussian.weighted_post_KL(mu0, Sig0inv, Siginv, x, w[m, :], reverse=True)
  fklw[m] = gaussian.weighted_post_KL(mu0, Sig0inv, Siginv, x, w[m, :], reverse=False)

if not os.path.exists('results/'):
  os.mkdir('results')
np.savez('results/results_'+nm+'_' + tr+'.npz', x=x, mu0=mu0, Sig0=Sig0, Sig=Sig, mup=mup, Sigp=Sigp, w=w,
                               muw=muw, Sigw=Sigw, rklw=rklw, fklw=fklw)

