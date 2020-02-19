from __future__ import print_function
import numpy as np
import scipy.linalg as sl
import pickle as pk 
import os, sys
hpc = True
if hpc:  sys.path.insert(1, os.path.join(sys.path[0], '/home/dm754/bayesian-coresets-private'))
import bayesiancoresets as bc
from scipy.optimize import minimize, nnls
import time

#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from mcmc import sampler
import gaussian

#computes the Laplace approximation N(mu, Sig) to the posterior with weights wts
def get_laplace(wts, Z, mu0, diag = False):
  trials = 10
  Zw = Z[wts>0, :]
  ww = wts[wts>0]
  while True:
    try:
      res = minimize(lambda mu : -log_joint(Zw, mu, ww)[0], mu0, jac=lambda mu : -grad_th_log_joint(Zw, mu, ww)[0,:])
    except:
      mu0 = mu0.copy()
      mu0 += np.sqrt((mu0**2).sum())*0.1*np.random.randn(mu0.shape[0])
      trials -= 1
      if trials <= 0:
        print('Tried laplace opt 10 times, failed')
        break
      continue
    break
  mu = res.x
  if diag:
    LSigInv = np.sqrt(-diag_hess_th_log_joint(Zw, mu, ww)[0,:])
    LSig = 1./LSigInv
  else:
    LSigInv = np.linalg.cholesky(-hess_th_log_joint(Zw, mu, ww)[0,:,:])
    LSig = sl.solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b = True, check_finite = False)
  return mu, LSig, LSigInv

dnm = sys.argv[1] #should be synth_lr / phishing / ds1 / synth_poiss / biketrips / airportdelays
alg = sys.argv[2] #should be GIGAO / GIGAR / RAND / PRIOR / SVI
ID = sys.argv[3] #just a number to denote trial #, any nonnegative integer

np.random.seed(int(ID))

lrdnms = ['synth_lr', 'phishing', 'ds1', 'synth_lr_large', 'phishing_large', 'ds1_large']
prdnms = ['synth_poiss', 'biketrips', 'airportdelays', 'synth_poiss_large', 'biketrips_large', 'airportdelays_large']
if dnm in lrdnms:
  from model_lr import *
else:
  from model_poiss import *

print('running ' + str(dnm)+ ' ' + str(alg)+ ' ' + str(ID))

if not os.path.exists('results/'):
  os.mkdir('results')

if not os.path.exists('results/'+dnm+'_samples.npy'):
  print('No MCMC samples found -- running STAN')
  #run sampler
  N_samples = 10000
  sampler(dnm, dnm in lrdnms, '../data/', 'results/', N_samples)


print('Loading dataset '+dnm)
Z, Zt, D = load_data('../data/'+dnm+'.npz')
print('Loading posterior samples for '+dnm)
samples = np.load('results/'+dnm+'_samples.npy')
#TODO FIX SAMPLER TO NOT HAVE TO DO THIS
samples = np.hstack((samples[:, 1:], samples[:, 0][:,np.newaxis]))

#fit a gaussian to the posterior samples 
#used for pihat computation for Hilbert coresets with noise to simulate uncertainty in a good pihat
mup = samples.mean(axis=0)
Sigp = np.cov(samples, rowvar=False)
LSigp = np.linalg.cholesky(Sigp)
LSigpInv = sl.solve_triangular(LSigp, np.eye(LSigp.shape[0]), lower=True, overwrite_b = True, check_finite = False)

#create the prior -- also used for the above purpose
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

###############################
## TUNING PARAMETERS ##
use_diag_laplace_w = False
M = 20
SVI_step_sched = lambda itr : 1./(1.+itr)
BPSVI_step_sched = lambda itr : 10./(1.+itr)
n_subsample_opt = 400
n_subsample_select = 1000
projection_dim = 100 #random projection dimension for Hilbert csts
pihat_noise = .75 #noise level (relative) for corrupting pihat
SVI_opt_itrs = 1500
BPSVI_opt_itrs = 1500
###############################

#get pihat via interpolation between prior/posterior + noise
#uniformly smooth between prior and posterior
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2.*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)

print('Building projectors')
sampler_optimal = lambda sz, w, pts : mup + np.random.randn(sz, mup.shape[0]).dot(LSigp.T)
sampler_realistic = lambda sz, w, pts : muhat + np.random.randn(sz, muhat.shape[0]).dot(LSighat.T)
def sampler_w(sz, w, pts):
  if pts.shape[0] == 0:
    w = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = get_laplace(w, pts, mu0, use_diag_laplace_w)
  if use_diag_laplace_w:
    return muw + np.random.randn(sz, muw.shape[0])*LSigw
  else:
    return muw + np.random.randn(sz, muw.shape[0]).dot(LSigw.T)

prj_optimal = bc.BlackBoxProjector(sampler_optimal, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_w = bc.BlackBoxProjector(sampler_w, projection_dim, log_likelihood, grad_z_log_likelihood)
 
print('Creating coresets object')
#create coreset construction objects
t0 = time.perf_counter()
giga_optimal = bc.HilbertCoreset(Z, prj_optimal)
gigao_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
giga_realistic = bc.HilbertCoreset(Z, prj_realistic)
gigar_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
unif = bc.UniformSamplingCoreset(Z)
unif_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs=SVI_opt_itrs, n_subsample_opt = n_subsample_opt, n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
sparsevi_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
bpsvi = bc.BatchPSVICoreset(Z, prj_w, opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt, step_sched = BPSVI_step_sched)
bpsvi_t_setup = time.perf_counter()-t0

algs = {'SVI': sparsevi, 
        'BPSVI': bpsvi, 
        'GIGAO': giga_optimal, 
        'GIGAR': giga_realistic, 
        'RAND': unif,
        'PRIOR': None}
coreset = algs[alg]
t0s = {'SVI' : sparsevi_t_setup,
       'BPSVI' : bpsvi_t_setup,
       'GIGAO' : gigao_t_setup,
       'GIGAR' : gigar_t_setup,
       'RAND' : unif_t_setup,
       'PRIOR' : 0.}

print('Building coresets via ' + alg)
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]
cputs = np.zeros(M+1)
cputs[0] = t0s[alg]
for m in range(1, M+1):
  print(str(m)+'/'+str(M))
  if alg != 'PRIOR':
    t0 = time.perf_counter()
    coreset.build(1, m)
    cputs[m] = time.perf_counter()-t0

    #record time and weights
    wts, pts, idcs = coreset.get()
    w.append(wts)
    p.append(pts)
  else:
    w.append(np.array([0.]))
    p.append(np.zeros((1, Z.shape[1])))
    
#get laplace approximations for each weight setting, and KL divergence to full posterior laplace approx mup Sigp
#used for a quick/dirty performance comparison without expensive posterior sample comparisons (e.g. energy distance)
mus_laplace = np.zeros((M+1, D))
Sigs_laplace = np.zeros((M+1, D, D))
rkls_laplace = np.zeros(M+1)
fkls_laplace = np.zeros(M+1)
print('Computing coreset Laplace approximation + approximate KL(posterior || coreset laplace)')
for m in range(M+1):
  mul, LSigl, LSiglInv = get_laplace(w[m], p[m], Z.mean(axis=0)[:D])
  mus_laplace[m,:] = mul
  Sigs_laplace[m,:,:] = LSigl.dot(LSigl.T)
  rkls_laplace[m] = gaussian.gaussian_KL(mul, Sigs_laplace[m,:,:], mup, LSigpInv.dot(LSigpInv.T))
  fkls_laplace[m] = gaussian.gaussian_KL(mup, Sigp, mul, LSiglInv.dot(LSiglInv.T))

#save results
f = open('results/'+dnm+'_'+alg+'_results_' +str(ID)+'.pk', 'wb')
res = (cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace)
pk.dump(res, f)
f.close()
