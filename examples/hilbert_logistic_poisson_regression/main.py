from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from inference import nuts, rhat, hmc
import time
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))

### FOR LOGISTIC REGRESSION
#from model_lr import *
#dnames = ['synth_lr', 'ds1', 'phishing']

# FOR POISSON REGRESSION
from model_poiss import *
dnames = ['synth_poiss', 'airportdelays', 'biketrips']


n_trials = 20
mcmc_steps = 5000 #total number of MH steps
mcmc_burn = 1000
projection_dim = 500 #random projection dimension
Ms = np.unique(np.logspace(0, 3, 10, dtype=int))


pbar = True #progress bar display flag
step_size_init = 0.001
n_leap = 15
target_a = 0.8
anms = ['GIGA', 'FW', 'RND']
algs = [bc.GIGACoreset, bc.FrankWolfeCoreset, bc.UniformSamplingHilbertCoreset]
mcmc_alg = hmc #nuts

for dnm in dnames:
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data('../data/'+dnm+'.npz')

  print('Computing Laplace approximation')
  t0 = time.time()
  res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0)[:D], jac=lambda mu : -grad_log_joint(Z, mu, np.ones(Z.shape[0])))
  mu = res.x
  cov = -np.linalg.inv(hess_log_joint(Z, mu))
  t_laplace = time.time() - t0
  var_scales=np.ones(cov.shape[0])
  #generate a sampler based on the laplace approx 
  sampler = lambda sz : np.atleast_2d(np.random.multivariate_normal(mu, cov, sz))

  cputs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  csizes = np.zeros((len(anms), n_trials, Ms.shape[0]))
  Fs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  cputs_full = np.zeros(n_trials)

  chains = np.zeros((n_trials, mcmc_steps, mu.shape[0]))
  for tr in range(n_trials):
    print('Trial ' + str(tr+1) +'/' + str(n_trials))

    print('Building tangent space projection')
    t0 = time.time()
    loglik = lambda th : np.hstack( [log_likelihood(Z, th[i,:])[:,np.newaxis] for i in range(th.shape[0])])
    mct = bc.MonteCarloFiniteTangentSpace(loglik, sampler, projection_dim)
    t_projection = time.time()-t0

    print('Running MCMC on the full dataset')
    logpZ = lambda th : log_joint(Z, th, np.ones(Z.shape[0]))
    glogpZ = lambda th : grad_log_joint(Z, th, np.ones(Z.shape[0]))
    mcmc_param_init = np.random.multivariate_normal(mu, cov)
    t0 = time.time()
    full_samples = mcmc_alg(logp = logpZ, gradlogp = glogpZ, 
                     x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, 
                     n_leapfrogs = n_leap, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a) 
    cputs_full[tr] = time.time()-t0
    chains[tr, :, :] = full_samples

    print('Running coreset construction / MCMC')
    for aidx, anm in enumerate(anms):
      print(anm +':')

      t0 = time.time()
      alg = algs[aidx](mct)
      t_setup = time.time() - t0

      t_alg = 0.
      for m in range(Ms.shape[0]):
        print('M = ' + str(Ms[m]) + ': coreset construction')
        #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
        t0 = time.time()
        alg.build(Ms[m])
        t_alg += time.time()-t0
        wts, idcs = alg.weights()
      
        logpZ = lambda th : log_joint(Z[idcs, :], th, wts)
        glogpZ = lambda th : grad_log_joint(Z[idcs, :], th, wts)
        mcmc_param_init = np.random.multivariate_normal(mu, cov)
        print('M = ' + str(Ms[m]) + ': MCMC')
        t0 = time.time()
        th_samples = mcmc_alg(logp=logpZ, gradlogp=glogpZ, 
                     x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, 
                     n_leapfrogs= n_leap, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a)
        t_alg_mcmc = time.time()-t0    

        print('M = ' + str(Ms[m]) + ': CPU times')
        cputs[aidx, tr, m] = t_laplace + t_projection + t_setup + t_alg + t_alg_mcmc
        print('M = ' + str(Ms[m]) + ': coreset sizes')
        csizes[aidx, tr, m] = wts.shape[0]
        print('M = ' + str(Ms[m]) + ': F norms')
        gcs = np.array([ grad_log_joint(Z[idcs, :], full_samples[i, :], wts) for i in range(full_samples.shape[0]) ])
        gfs = np.array([ grad_log_joint(Z, full_samples[i, :], np.ones(Z.shape[0])) for i in range(full_samples.shape[0]) ])
        Fs[aidx, tr, m] = (((gcs - gfs)**2).sum(axis=1)).mean()
  #print(rhat(chains))
  if not os.path.exists('results/'):
    os.mkdir('results')  
  np.savez_compressed('results/'+dnm+'_results.npz', Ms=Ms, Fs=Fs, cputs=cputs, cputs_full=cputs_full, csizes=csizes, anms=anms)
