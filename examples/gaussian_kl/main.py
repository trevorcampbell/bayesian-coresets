import numpy as np
from exact import *
from stochastic import *
from gaussian import *

opt_itrs = 1000
M = 20
N = 20
n_samples = 1000
mu0 = np.zeros(2)
Sig0 = np.eye(2)
Sig = np.eye(2)
th = np.ones(2)
x = np.random.multivariate_normal(th, Sig, N)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
mup, Sigp = weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
Sigpinv = np.linalg.inv(Sigp)

np.random.seed()

erl1 = EGL1Reverse(x, mu0, Sig0, Sig)
efl1 = EGL1Forward(x, mu0, Sig0, Sig)
erg = EGGreedyReverse(x, mu0, Sig0, Sig)
efg = EGGreedyForward(x, mu0, Sig0, Sig)

srl1 = SGL1Reverse(x, mu0, Sig0, Sig, n_samples)
sfl1 = SGL1Forward(x, mu0, Sig0, Sig, n_samples)
srg = SGGreedyReverse(x, mu0, Sig0, Sig, n_samples)
sfg = SGGreedyForward(x, mu0, Sig0, Sig, n_samples)

w_erl1 = np.zeros((M+1, x.shape[0]))
w_efl1 = np.zeros((M+1, x.shape[0]))
w_erg = np.zeros((M+1, x.shape[0]))
w_efg = np.zeros((M+1, x.shape[0]))
w_srl1 = np.zeros((M+1, x.shape[0]))
w_sfl1 = np.zeros((M+1, x.shape[0]))
w_srg = np.zeros((M+1, x.shape[0]))
w_sfg = np.zeros((M+1, x.shape[0]))


algs = [erl1, efl1, erg, efg, srl1, sfl1, srg, sfg]
ws = [w_erl1, w_efl1, w_erg, w_efg, w_srl1, w_sfl1, w_srg, w_sfg]
nms = ['ERL1', 'EFL1', 'ERG', 'EFG', 'SRL1', 'SFL1', 'SRG', 'SFG']

algs = [erl1, efl1, erg, efg]
ws = [w_erl1, w_efl1, w_erg, w_efg]
nms = ['ERL1', 'EFL1', 'ERG', 'EFG']


for w, nm, alg in zip(ws, nms, algs):

  for m in range(1, M+1):
    alg.build(m)
    w[m, :] = alg.weights()

  muw = np.zeros((M+1, mu0.shape[0]))
  Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
  rklw = np.zeros(M+1)
  fklw = np.zeros(M+1)
  for m in range(M+1):
    muw[m, :], Sigw[m, :, :] = weighted_post(mu0, Sig0inv, Siginv, x, w[m, :])
    rklw[m] = weighted_post_KL(mu0, Sig0inv, Siginv, x, w[m, :], reverse=True)
    fklw[m] = weighted_post_KL(mu0, Sig0inv, Siginv, x, w[m, :], reverse=False)
  
  np.savez('results_'+nm+'.npz', x=x, mu0=mu0, Sig0=Sig0, Sig=Sig, mup=mup, Sigp=Sigp, w=w,
                                 muw=muw, Sigw=Sigw, rklw=rklw, fklw=fklw)
