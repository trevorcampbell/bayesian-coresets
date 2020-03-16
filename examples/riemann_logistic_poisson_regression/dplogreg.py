from __future__ import print_function
import numpy as np
import scipy.linalg as sl
import pickle as pk 
import os, sys
from scipy.optimize import minimize, nnls
import time 

# linearize for slurm job array submission
def linearize():
  args_dict = dict()
  c = -1
  for nmult in [100.]:
    for i0 in [0.05]: #[500., 1000.]:
      for ID in range(0, 1):
        for alg in ["DPBPSVI"]: #, "BPSVI", "SVI"]: #["RAND", "PRIOR", "GIGAO", "GIGAR", "BPSVI", "SVI"]:
          for dnm in ['ds1.100']: #['mnist2class_test']: #['ds1.100']: #['mnist2class_test']: #['fma', 'ds1.100', 'mnist2class_test']:
            c += 1
            args_dict[c] = (dnm, alg, ID, i0, nmult)
  return args_dict

mapping = linearize()
print('mapping : ', len(mapping))
#exit()
#dnm, alg, ID, i0, nmult = mapping[int(sys.argv[1])]
dnm, alg, ID, i0, nmult = mapping[0]
hpc = True
if hpc: sys.path.insert(1, os.path.join(sys.path[0], '/home/dm754/bayesian-coresets-private'))
import bayesiancoresets as bc

#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from mcmc_noXt import sampler
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
    LSig = sl.solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b=True, check_finite=False)
  return mu, LSig, LSigInv

np.random.seed(int(ID))

lrdnms = ['synth_lr', 'phishing', 'ds1', 'synth_lr_large', 'phishing_large', 'ds1_large', 'ds1.100', 'mnist2class_test', 'fma']
largescale = ['ds1.100', 'mnist2class_test', 'fma']

if dnm in lrdnms: from model_lr import *
else: from model_poiss import *

print('running ' + str(dnm)+ ' ' + str(alg)+ ' ' + str(ID))

###############################
## TUNING PARAMETERS ##
if dnm in largescale:
  use_diag_laplace_w = True
else:
  use_diag_laplace_w = False
M = 15
SVI_step_sched = lambda itr : i0/(1.+itr)
BPSVI_step_sched = lambda m: lambda itr : (50. + 50*(1.-m/M))/(1.+itr) # make step schedule potentially dependent on coreset size
DPBPSVI_step_sched = BPSVI_step_sched
n_subsample_opt = 200
n_subsample_select = 1000
projection_dim = 100 #random projection dimension for Hilbert csts
pihat_noise = .75 #noise level (relative) for corrupting pihat
SVI_opt_itrs = 500
BPSVI_opt_itrs = 500
DPBPSVI_opt_itrs = BPSVI_opt_itrs
###############################

print('Loading dataset '+dnm)
Z, Zt, D = load_data('../data/'+dnm+'.npz')
if not os.path.exists('results/'):
  os.mkdir('results')
print('shape : ', Z.shape)
  
if not os.path.exists('results/'+dnm+'_samples.npy'):
  N_samples=10000
  if dnm in largescale: # sample using laplace approximation on full dataset
    print('sampling using laplace with using diag : ', use_diag_laplace_w) 
    mup_laplace, LSigp_laplace, LSigpInv_laplace = get_laplace(np.ones(Z.shape[0]), Z, Z.mean(axis=0)[:D], use_diag_laplace_w)
    if use_diag_laplace_w:
      samples_laplace = mup_laplace + np.random.randn(N_samples, mup_laplace.shape[0])*LSigp_laplace
    else:
      samples_laplace = mup_laplace + np.random.randn(N_samples, mup_laplace.shape[0]).dot(LSigp_laplace.T)
    np.save(os.path.join('results/'+dnm+'_samples.npy'), samples_laplace)
  else: # use stan sampler
    print('No MCMC samples found -- running STAN')
    sampler(dnm, dnm in lrdnms, '../data/', 'results/', N_samples)

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
print('posterior fitting done')

#create the prior -- also used for the above purpose
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

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

def gen_inits(m, thetas):
  np.random.seed(m)
  X = 5*np.random.multivariate_normal(mu0, Sig0, m)
  ps = 1.0/(1.0+np.exp(-(X*thetas).sum(axis=1)))
  y = (np.random.rand(m) <= ps).astype(int)
  y[y==0] = -1
  return y[:, np.newaxis]*X

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

t0 = time.perf_counter()
dpbpsvi = bc.DiffPrivBatchPSVICoreset(Z, prj_w, opt_itrs = DPBPSVI_opt_itrs, n_subsample_opt = n_subsample_opt, step_sched = DPBPSVI_step_sched,
                                     init_sampler = sampler_w, gen_inits = gen_inits, noise_multiplier = nmult)
dpbpsvi_t_setup = time.perf_counter()-t0

algs = {'SVI': sparsevi, 
        'BPSVI': bpsvi, 
        'DPBPSVI': dpbpsvi,
        'GIGAO': giga_optimal, 
        'GIGAR': giga_realistic, 
        'RAND': unif,
        'PRIOR': None}
coreset = algs[alg]
t0s = {'SVI' : sparsevi_t_setup,
       'BPSVI' : bpsvi_t_setup,
       'DPBPSVI': dpbpsvi_t_setup,
       'GIGAO' : gigao_t_setup,
       'GIGAR' : gigar_t_setup,
       'RAND' : unif_t_setup,
       'PRIOR' : 0.}

print('Building coresets via ' + alg)
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]
cputs = np.zeros(M+1)
cputs[0] = t0s[alg]

def build_per_m(m): # consturction in parallel for different coreset sizes used in BPSVI
  t0 = time.perf_counter()
  coreset.build(1, m)
  print('m = ', m, ' done')
  return coreset.get(), time.perf_counter()-t0

if alg in ['BPSVI', 'QDPBPSVI']:
  from multiprocessing import Pool
  pool = Pool(processes=80)
  res = pool.map(build_per_m, range(1, M+1))
  i=1
  for (wts, pts, _), cput in res:
    w.append(wts)
    p.append(pts)
    cputs[i] = cput
    i+=1
else:
  for m in range(1, M+1):
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
  print('shapes of w and p : ', w[m].shape, p[m].shape)
  input()
  mul, LSigl, LSiglInv = get_laplace(w[m], p[m], Z.mean(axis=0)[:D])
  mus_laplace[m,:] = mul
  Sigs_laplace[m,:,:] = LSigl.dot(LSigl.T)
  print('rkld')
  rkls_laplace[m] = gaussian.gaussian_KL(mul, Sigs_laplace[m,:,:], mup, LSigpInv.dot(LSigpInv.T))
  #fkls_laplace[m] = gaussian.gaussian_KL(mup, Sigp, mul, LSiglInv.dot(LSiglInv.T))
  print('reverse kl : ', rkls_laplace[m])
#save results
f = open('results/dp_adaptive_'+dnm+'_'+alg+'_results_'+str(i0)+'_nmult'+str(nmult)+'_'+str(ID)+'.pk', 'wb')
res = (cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace)#, fkls_laplace)
pk.dump(res, f)
f.close()



