import numpy as np
from exact import *
from stochastic import *
from sampling import *
from gaussian import *
import bayesiancoresets as bc
from copy import deepcopy
import os
from scipy.stats import multivariate_normal



np.random.seed(1)

M = 20
N = 1000
d = 30
n_samples = 1000
trials = np.arange(100)
mu0 = np.zeros(d)
Sig0 = np.eye(d)
Sig = np.eye(d)
th = np.ones(d)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
opt_itrs = 1000
giga_proj_dim = 200
pihat_noise =0.15

for t in trials:
  #gen data
  x = np.random.multivariate_normal(th, Sig, N)
  mup, Sigp = weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
  Sigpinv = np.linalg.inv(Sigp)

  #create coreset objects
  erl1 = EGL1Reverse(x, mu0, Sig0, Sig, 20., 1., opt_itrs, scaled=True)
  erl1u = EGL1Reverse(x, mu0, Sig0, Sig, 1., 1., opt_itrs, scaled=False)
  #efl1 = EGL1Forward(x, mu0, Sig0, Sig, scaled=scaled)
  erg = EGGreedyReverse(x, mu0, Sig0, Sig, 1., 1., opt_itrs)
  #efg = EGGreedyForward(x, mu0, Sig0, Sig, scaled=scaled)
  ercg = EGCorrectiveGreedyReverse(x, mu0, Sig0, Sig, 10., 1., opt_itrs)
  
  #srl1 = SGL1Reverse(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #sfl1 = SGL1Forward(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #srg = SGGreedyReverse(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #sfg = SGGreedyForward(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  
  #sgs = SGS(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #egs = EGS(x, mu0, Sig0, Sig, scaled=scaled)
  egus = EGUS(x, mu0, Sig0, Sig, 10., 1., opt_itrs)
  
  #algs = [erl1, efl1, erg, efg, srl1, sfl1, srg, sfg, sgs, egs, egus]
  #nms = ['ERL1', 'EFL1', 'ERG', 'EFG', 'SRL1', 'SFL1', 'SRG', 'SFG', 'SGS', 'EGS', 'EGUS']

  #get good projection samples from smoothed true posterior
  samps_good = np.random.multivariate_normal(mup, 9*Sigp, giga_proj_dim)
  #get bad samples from a noisy pihat via interpolation between prior/posterior + noise
  #uniformly smooth between prior and posterior
  U = np.random.rand()
  muhat = U*mup + (1.-U)*mu0
  Sighat = U*Sigp + (1.-U)*Sig0
  #now corrupt the smoothed pihat
  muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
  Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))

  ##super bad corruption
  #muhat = 1000*np.random.randn(d)
  #Sighat = 1000*np.random.randn(d, d)
  #Sighat = Sighat.T.dot(Sighat)

  samps_bad = np.ones((giga_proj_dim, muhat.shape[0])) #np.random.multivariate_normal(muhat, Sighat, giga_proj_dim) 

  #compute log likelihood feature vectors for both
  lls_bad = np.zeros((x.shape[0], giga_proj_dim))
  lls_good = np.zeros((x.shape[0], giga_proj_dim))
  for i in range(x.shape[0]):
    lls_bad[i, :] = multivariate_normal.logpdf(samps_bad, x[i,:], Sig)
    lls_good[i, :] = multivariate_normal.logpdf(samps_good, x[i,:], Sig)
  lls_bad -= lls_bad.mean(axis=1)[:,np.newaxis]
  lls_good -= lls_good.mean(axis=1)[:,np.newaxis]

  #corrs = (((lls_bad - lls_bad.mean(axis=0))*(lls_good-lls_good.mean(axis=0))).mean(axis=0)/lls_bad.std(axis=0)/lls_good.std(axis=0))
  #print('bad/good correlation: ' + str( corrs.mean()) + ' +/- ' + str(corrs.std())) 
  #continue


  giga_bad = bc.GIGACoreset(lls_bad)
  giga_good = bc.GIGACoreset(lls_good)
 
  algs = [erl1, erl1u, erg, ercg, egus, giga_bad, giga_good]
  nms = ['ERL1', 'ERL1U', 'ERG', 'ERCG', 'EGUS', 'GIGAB', 'GIGAG']

  #algs = [egus]
  #nms = ['EGUS']

  algs = [giga_bad, giga_good]
  nms = ['GIGAB', 'GIGAG']

  #build coresets
  for nm, alg in zip(nms, algs):
    w = np.zeros((M+1, x.shape[0]))
    w_opt = np.zeros((M+1, x.shape[0]))
    for m in range(1, M+1):
      print('trial: ' + str(t+1)+'/'+str(trials.shape[0])+' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      alg.build(m)
      w[m, :] = alg.weights()
      tmpalg = deepcopy(alg)
      tmpalg.optimize()
      w_opt[m, :] = tmpalg.weights()
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
