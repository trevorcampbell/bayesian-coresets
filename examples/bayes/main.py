from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from nuts import nuts, rhat
import time


## FOR LOGISTIC REGRESSION
from model_lr import *
dnames = ['synth', 'ds1', 'phishing']
fldr = 'lr'

## FOR POISSON REGRESSION
#from model_poiss import *
#dnames = ['synth', 'airportdelays', 'biketrips']
#fldr = 'poiss'

n_trials = 50
mcmc_steps = 5000 #total number of MH steps
mcmc_burn = 1000
projection_dim = 500 #random projection dimension
Ms = np.unique(np.logspace(0, 3, 10, dtype=int))
pbar = True #NUTS progress bar display flag
step_size_init = 0.01
target_a = 0.65
anms = ['GIGA', 'FW', 'RND']


for dnm in dnames:
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data(fldr+'/'+dnm+'.npz')

  print('Computing Laplace approximation')
  t0 = time.time()
  res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0)[:D], jac=lambda mu : -grad_log_joint(Z, mu, np.ones(Z.shape[0])))
  mu = res.x
  cov = -np.linalg.inv(hess_log_joint(Z, mu))
  t_laplace = time.time() - t0
  var_scales = np.diag(cov)

  cputs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  csizes = np.zeros((len(anms), n_trials, Ms.shape[0]))
  Fs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  Fs_full = np.zeros(n_trials)
  cputs_full = np.zeros(n_trials)

  chains = np.zeros((n_trials, mcmc_steps, mu.shape[0]))
  for tr in range(n_trials):
    print('Trial ' + str(tr+1) +'/' + str(n_trials))

    print('Computing random projection')
    t0 = time.time()
    proj = bc.ProjectionF(Z, grad_log_likelihood, projection_dim, lambda : np.random.multivariate_normal(mu, cov)) 
    vecs = proj.get()
    t_projection = time.time()-t0

    print('Running NUTS on the full dataset')
    logpZ = lambda th : log_joint(Z, th, np.ones(Z.shape[0]))
    glogpZ = lambda th : grad_log_joint(Z, th, np.ones(Z.shape[0]))
    mcmc_param_init = np.random.multivariate_normal(mu, cov)
    t0 = time.time()
    full_samples = nuts(logp = logpZ, gradlogp = glogpZ, 
                     x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a) 
    cputs_full[tr] = time.time()-t0
    Fs_full[tr] = 0. #always 0, just doing this to make later code simpler
    chains[tr, :, :] = full_samples
    
    print('Running coreset construction / NUTS')
    for aidx, anm in enumerate(anms):
      print(anm +':')

      t0 = time.time()
      alg = None
      if 'GIGA' in anm:
        alg = bc.GIGA(vecs)
      elif anm == 'FW':
        alg = bc.FrankWolfe(vecs)
      else:
        alg = bc.RandomSubsampling(vecs) 
      t_setup = time.time() - t0

      t_alg = 0.
      for m in range(Ms.shape[0]):
        print('M = ' + str(Ms[m]) + ': coreset construction')
        #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
        t0 = time.time()
        alg.run(Ms[m])
        t_alg += time.time()-t0
        wts = alg.weights()
        idcs = wts > 0
      
        logpZ = lambda th : log_joint(Z[idcs, :], th, wts[idcs])
        glogpZ = lambda th : grad_log_joint(Z[idcs, :], th, wts[idcs])
        mcmc_param_init = np.random.multivariate_normal(mu, cov)
        print('M = ' + str(Ms[m]) + ': NUTS')
        t0 = time.time()
        th_samples = nuts(logp=logpZ, gradlogp=glogpZ, 
                     x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a)
        t_alg_mcmc = time.time()-t0    

        print('M = ' + str(Ms[m]) + ': CPU times')
        cputs[aidx, tr, m] = t_laplace + t_projection + t_setup + t_alg + t_alg_mcmc
        print('M = ' + str(Ms[m]) + ': coreset sizes')
        csizes[aidx, tr, m] = (wts>0).sum()
        print('M = ' + str(Ms[m]) + ': F norms')
        gcs = np.array([ grad_log_joint(Z[idcs, :], full_samples[i, :], wts[idcs]) for i in range(full_samples.shape[0]) ])
        gfs = np.array([ grad_log_joint(Z, full_samples[i, :], np.ones(Z.shape[0])) for i in range(full_samples.shape[0]) ])
        Fs[aidx, tr, m] = (((gcs - gfs)**2).sum(axis=1)).mean()
  print(rhat(chains))
  np.savez_compressed(fldr+'/'+dnm+'_results.npz', Ms=Ms, Fs=Fs, Fs_full=Fs_full, cputs=cputs, cputs_full=cputs_full, csizes=csizes, anms=anms)
