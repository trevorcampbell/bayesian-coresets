from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
import time

def gaussian_KL(mu0, Sig0, mu1, Sig1):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.linalg.solve(Sig1, mu1-mu0))
  t3 = np.linalg.slogdet(Sig1)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])


#(wts[m,:][:,np.newaxis]*Z).mean(axis=0)[:D]
def get_laplace(wts, Z, mu0)
  res = minimize(lambda mu : -log_joint(Z, mu, wts[m, :]), mu0, jac=lambda mu : -grad_log_joint(Z, mu, wts[m,:]))
  mu = res.x
  Sig = -np.linalg.inv(hess_log_joint(Z, mu))
  return mu, Sig

def riemann_select(Z, w, muw, Sigw, n_samples):
  #take samples for empirical correlation estimation 
  samps = np.random.multivariate_normal(muw, Sigw, n_samples)
  #compute log likelihoods
  lls = np.zeros((Z.shape[0], n_samples))
  for i in range(n_samples):
    lls[:, i] = log_likelihood(Z, samps[i,:])
  #subtract off the mean
  lls -= lls.mean(axis=1)[:, np.newaxis]
  #compute residual error
  residuals = lls.sum(axis=0) - w.dot(lls) 
  #get std dev of lls
  stds = lls.std(axis=1)
  #compute correlations (w/o normalizing residual, since it doesn't affect selection)
  corrs = (lls*residuals).mean(axis=1)/stds
  #for data in the active set, just look at abs(corr)
  corrs[w>0] = np.fabs(corrs[w>0])
  return corrs.argmax()
  
def grad_line(Z, w, one_n, samps, beta, alpha):
  lls = np.zeros((Z.shape[0], n_samples))
  for i in range(n_samples):
    lls[:, i] = log_likelihood(Z, samps[i,:])
  #subtract off the mean
  lls -= lls.mean(axis=1)[:, np.newaxis]
  wab = beta*(w+alpha*one_n)
  #compute gradients
  one_f = lls.sum(axis=0)
  wab_f = wab.dot(lls)
  dKLdb = -1./beta*(wab_f*(one_f-wab_f)).mean()
  dKLda = -beta*(lls[n,:]*(one_f-wab_f)).mean()
  return dKLda, dKLdb

def riemann_optimize_line(Z, w, n, muw, Sigw, n_samples):
  one_n = np.zeros(Z.shape[0])
  one_n[n] = 1.
  samps = np.random.multivariate_normal(muw, Sigw, n_samples)
  grad_line(...)

def grad_full(Z, w, samps, active_idcs):
  #compute log likelihoods
  lls = np.zeros((Z.shape[0], n_samples))
  for i in range(n_samples):
    lls[:, i] = log_likelihood(Z, samps[i,:])
  #subtract off the mean
  lls -= lls.mean(axis=1)[:, np.newaxis]
  #compute residual error
  residuals = lls.sum(axis=0) - w.dot(lls) 
  #compute gradient
  dKLdw = (lls[active_idcs, :]*residuals).mean(axis=1)
  return dKLdw


def riemann_optimize_full(Z, w, n, muw, Sigw, n_samples):
  active_idcs = w>0
  active_idcs[n] = True

  #take samples for empirical correlation estimation 
  samps = np.random.multivariate_normal(muw, Sigw, n_samples)
  grad_full(...)

fldr = sys.argv[1]
dnm = sys.argv[2]
alg = sys.argv[3]
ID = sys.argv[4]

if fldr == 'lr':
  from model_lr import *
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data('lr/'+dnm+'.npz')
  print('Loading posterior samples for '+dnm)
  samples = np.load('lr/'+dnm+'_samples.npz')
else:
  from model_poiss import *
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data('poiss/'+dnm+'.npz')
  print('Loading posterior samples for '+dnm)
  samples = np.load('poiss/'+dnm+'_samples.npz')

#fit a gaussian to the posterior samples, used for pihat smoothing
mup = samples.mean(axis=0)
Sigp = np.cov(samples, rowvar=False)
#create the prior -- also used for pihat smoothing
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

Ms = [1, 2, 5, 10, 20, 50, 100]
projection_dim = 500 #random projection dimension
pihat_noise = 0.15

#initialize memory for coreset weights, laplace approx, kls
wts = np.zeros((len(Ms), Z.shape[0]))
cputs = np.zeros(len(Ms))
print('Building coresets via ' + alg)
t0 = time.clock()
if alg == 'hilbert' or 'hilbert_corr':
  #get pihat via interpolation between prior/posterior + noise
  U = np.random.rand()
  muhat = U*mup + (1.-U)*mu0
  Sighat = U*Sigp + (1.-U)*Sig0
  muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
  Sighat *= np.exp(2*pihat_noise*np.random.randn())
  #take pihat samples
  proj_samps = np.random.multivariate_normal(muhat, Sighat, projection_dim)
  #compute random projection
  lls = np.zeros((Z.shape[0], projection_dim))
  for i in range(proj_samps.shape[0]):
    lls[:, i] = log_likelihood(Z, proj_samps[i, :])
  lls -= lls.mean(axis=1)[:,np.newaxis]
  #Build coreset via GIGA
  giga = bc.GIGACoreset(lls)
  for m in range(len(Ms)):
    print(str(m+1)+'/'+str(len(Ms)))
    giga.build(Ms[m])
    if alg == 'hilbert_corr':
      giga.optimize() 
    #record time
    cputs[m] = time.clock()-t0
    wts[m, :] = giga.weights()
elif alg == 'riemann' or 'riemann_corr':
  #normal dist for approx piw sampling; will use laplace throughout
  muw = np.zeros(mup.shape[0])
  Sigw = np.eye(mup.shape[0])
  w = np.zeros(Z.shape[0])
  for m in range(len(Ms)):
    #build up to Ms[m] one point at a time
    for j in range(Ms[m]-Ms[m-1] if m>0 else Ms[m]):
      n = riemann_select(Z, w, muw, Sigw)
      if alg == 'riemann_corr':
        w, muw, Sigw = riemann_optimize_full(Z, w, n, muw, Sigw)
      else:
        w, muw, Sigw = riemann_optimize_line(Z, w, n, muw, Sigw)
    wts[m, :] = w.copy()
    #record time
    cputs[m] = time.clock()-t0
elif alg == 'uniform':
  print(str(1)+'/'+str(len(Ms)))
  wts[0, :] = np.random.multinomial(Ms[0], np.ones(Z.shape[0])/float(Z.shape[0]))
  cputs[0] = time.clock() - t0
  for m in range(1, len(Ms)):
    print(str(m+1)+'/'+str(len(Ms)))
    wts[m, :] = wts[m-1, :] + np.random.multinomial(Ms[m]-Ms[m-1], np.ones(Z.shape[0])/float(Z.shape[0]))
    #record time
    cputs[m] = time.clock() - t0

#get laplace
mus_laplace = np.zeros((len(Ms), D))
Sigs_laplace = np.zeros((len(Ms), D, D))
kls_laplace = np.zeros(len(Ms))
print('Computing coreset Laplace approximation + approximate KL(posterior || coreset laplace)')
for m in range(len(Ms)):
  mul, Sigl = get_laplace(wts[m,:], Z)
  mus_laplace[m,:] = mul
  Sigs_laplace[m,:,:] = Sigl
  kls_laplace[m] = gaussian_KL(mup, Sigp, mus_laplace[m,:], Sigs_laplace[m,:,:])

#save results
np.savez(fldr+'_'+dnm+'_results_'+str(ID)+'.npz', cputs=cputs, wts=wts, Ms=Ms, mus=mus_laplace, Sigs=Sigs_laplace, kls=kls_laplace)


