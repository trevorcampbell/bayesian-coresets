import numpy as np
import bayesiancoresets as bc
from copy import deepcopy, copy
import os
from scipy.stats import multivariate_normal


def gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples):
  return -x.shape[1]/2*np.log(2*np.pi) - 1./2.*logdetSig - 1./2.*(xSiginvx[:, np.newaxis] - 2.*np.dot(xSiginv, samples.T) + (np.dot(samples, Siginv)*samples).sum(axis=1))
 
def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post(th0, Sig0inv, Siginv, x, w): 
  Sigp = np.linalg.inv(Sig0inv + w.sum()*Siginv)
  mup = np.dot(Sigp,  np.dot(Sig0inv,th0) + np.dot(Siginv, (w[:, np.newaxis]*x).sum(axis=0)))
  return mup, Sigp

def weighted_post_KL(th0, Sig0inv, Siginv, x, w, reverse=True):
  muw, Sigw = weighted_post(th0, Sig0inv, Siginv, x, w)
  mup, Sigp = weighted_post(th0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
  if reverse:
    return gaussian_KL(muw, Sigw, mup, np.linalg.inv(Sigp))
  else:
    return gaussian_KL(mup, Sigp, muw, np.linalg.inv(Sigw))


np.random.seed(1)

M = 20
N = 1000
d = 30
n_samples = 1000
trials = np.arange(100)
mu0 = np.zeros(d)
Sig0 = np.eye(d)
Sig = np.eye(d)
SigL = np.linalg.cholesky(Sig)
th = np.ones(d)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
SigLInv = np.linalg.inv(SigL)
opt_itrs = 1000
proj_dim = 100
pihat_noise =0.15

for t in trials:
  #generate data and compute true posterior
  x = np.random.multivariate_normal(th, Sig, N)
  mup, Sigp = weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
  Sigpinv = np.linalg.inv(Sigp)

  #compute constants for log likelihood function
  xSiginv = x.dot(Siginv)
  xSiginvx = (xSiginv*x).sum(axis=1)
  logdetSig = np.linalg.slogdet(Sig)[1]

  log_likelihood = lambda samples : gaussian_potentials(Siginv, xSiginvx, xSiginv, logdetSig, x, samples)

  #create tangent space for well-tuned Hilbert coreset alg
  T_true = bc.MonteCarloFiniteTangentSpace(log_likelihood, lambda sz : np.random.multivariate_normal(mup, 9*Sigp, sz), proj_dim)

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
    muw, Sigw = weighted_post(mu0, Sig0inv, Siginv, x, w)
    nu = (x - muw).dot(SigLInv.T)
    Psi = np.dot(SigLInv, np.dot(Sigw, SigLInv.T))

    nu = np.hstack((nu.dot(np.linalg.cholesky(Psi)), 0.25*np.sqrt(np.trace(np.dot(Psi.T, Psi)))*np.ones(nu.shape[0])[:,np.newaxis]))
    
    return bc.FixedFiniteTangentSpace(nu, wts, idcs)
    
   
  #create coreset construction objects
  riemann_one = bc.SparseVICoreset(x.shape[0], tangent_space_factory, step_size = 1., update_single=True)
  riemann_full = bc.SparseVICoreset(x.shape[0], tangent_space_factory, step_size = 1., update_single=False)
  giga_true = bc.GIGACoreset(T_true)
  giga_noisy = bc.GIGACoreset(T_noisy)
  unif = bc.UniformSamplingCoreset(T_true) #tangent space unimportant here
 
  algs = [riemann_one, riemann_full, giga_true, giga_noisy, unif]
  nms = ['SVI1', 'SVIF', 'GIGAT', 'GIGAN', 'RAND']

  #build coresets
  for nm, alg in zip(nms, algs):
    w = np.zeros((M+1, x.shape[0]))
    w_opt = np.zeros((M+1, x.shape[0]))
    for m in range(1, M+1):
      print('trial: ' + str(t+1)+'/'+str(trials.shape[0])+' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      alg.build(m)
      wts, idcs = alg.weights()
      w[m, idcs] = wts
      tmpalg = copy(alg)
      tmpalg.optimize()
      wts, idcs = tmpalg.weights()
      w_opt[m, idcs] = wts
      #print('reverse KL: ' + str(weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)))
      #print('reverse KL opt: ' + str(weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)))

    muw = np.zeros((M+1, mu0.shape[0]))
    Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
    rklw = np.zeros(M+1)
    fklw = np.zeros(M+1)
    for m in range(M+1):
      muw[m, :], Sigw[m, :, :] = weighted_post(mu0, Sig0inv, Siginv, x, w[m, :])
      rklw[m] = weighted_post_KL(mu0, Sig0inv, Siginv, x, w[m, :], reverse=True)
      fklw[m] = weighted_post_KL(mu0, Sig0inv, Siginv, x, w[m, :], reverse=False)
    muw_opt = np.zeros((M+1, mu0.shape[0]))
    Sigw_opt = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
    rklw_opt = np.zeros(M+1)
    fklw_opt = np.zeros(M+1)
    for m in range(M+1):
      muw_opt[m, :], Sigw_opt[m, :, :] = weighted_post(mu0, Sig0inv, Siginv, x, w_opt[m, :])
      rklw_opt[m] = weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=True)
      fklw_opt[m] = weighted_post_KL(mu0, Sig0inv, Siginv, x, w_opt[m, :], reverse=False)

    if not os.path.exists('results/'):
      os.mkdir('results')
    np.savez('results/results_'+nm+'_' + str(t)+'.npz', x=x, mu0=mu0, Sig0=Sig0, Sig=Sig, mup=mup, Sigp=Sigp, w=w, w_opt=w_opt,
                                   muw=muw, Sigw=Sigw, rklw=rklw, fklw=fklw,
                                   muw_opt=muw_opt, Sigw_opt=Sigw_opt, rklw_opt=rklw_opt, fklw_opt=fklw_opt)
