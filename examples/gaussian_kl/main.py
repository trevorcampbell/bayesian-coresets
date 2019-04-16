import numpy as np
from exact import *
from stochastic import *
from sampling import *
from gaussian import *
import bayesiancoresets as bc



np.random.seed(1)

opt_itrs = 1000
M = 50
N = 100
n_samples = 1000
n_trials = 100
mu0 = np.zeros(2)
Sig0 = np.eye(2)
Sig = np.eye(2)
th = np.ones(2)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
scaled = True


for t in range(n_trials):
  #gen data
  x = np.random.multivariate_normal(th, Sig, N)
  mup, Sigp = weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
  Sigpinv = np.linalg.inv(Sigp)

  #create coreset objects
  erl1 = EGL1Reverse(x, mu0, Sig0, Sig, scaled=scaled)
  #efl1 = EGL1Forward(x, mu0, Sig0, Sig, scaled=scaled)
  erg = EGGreedyReverse(x, mu0, Sig0, Sig)
  #efg = EGGreedyForward(x, mu0, Sig0, Sig, scaled=scaled)
  
  #srl1 = SGL1Reverse(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #sfl1 = SGL1Forward(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #srg = SGGreedyReverse(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #sfg = SGGreedyForward(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  
  #sgs = SGS(x, mu0, Sig0, Sig, n_samples, scaled=scaled)
  #egs = EGS(x, mu0, Sig0, Sig, scaled=scaled)
  egus = EGUS(x, mu0, Sig0, Sig)
  
  #algs = [erl1, efl1, erg, efg, srl1, sfl1, srg, sfg, sgs, egs, egus]
  #nms = ['ERL1', 'EFL1', 'ERG', 'EFG', 'SRL1', 'SFL1', 'SRG', 'SFG', 'SGS', 'EGS', 'EGUS']
  
  algs = [erl1, erg, egus]
  nms = ['ERL1', 'ERG', 'EGUS']

  #build coresets
  for nm, alg in zip(nms, algs):
    w = np.zeros((M+1, x.shape[0]))
    w_opt = np.zeros((M+1, x.shape[0]))
    for m in range(1, M+1):
      print('trial: ' + str(t+1)+'/'+str(n_trials)+' alg: ' + nm + ' ' + str(m) +'/'+str(M))
      alg.build(m)
      w[m, :] = alg.weights()
      alg.optimize()
      w_opt[m, :] = alg.weights()

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

    np.savez('results_'+nm+'_'+('scaled' if scaled else 'unscaled') + '_' + str(t)+'.npz', x=x, mu0=mu0, Sig0=Sig0, Sig=Sig, mup=mup, Sigp=Sigp, w=w, w_opt=w_opt,
                                   muw=muw, Sigw=Sigw, rklw=rklw, fklw=fklw,
                                   muw_opt=muw_opt, Sigw_opt=Sigw_opt, rklw_opt=rklw_opt, fklw_opt=fklw_opt)
