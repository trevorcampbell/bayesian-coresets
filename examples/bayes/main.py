from __future__ import print_function
import numpy as np
from inference import mh
import bayesiancoresets as bc
from scipy.optimize import minimize
import time


## FOR LOGISTIC REGRESSION
from model_lr import *
dnames = ['synth', 'ds1', 'phishing']
fldr = 'lr'

## FOR POISSON REGRESSION
#from model_poiss import *
#dnames = ['synth', 'airportdelays', 'biketrips']
#fldr = 'poiss'

mh_steps = 50000 #total number of MH steps
mh_thin = 5 #thinning factor
mh_target = 0.234 #target acceptance rate
mh_step_var_init = 0.1 #initial step variance
n_samples = mh_steps / 2 / mh_thin #number of output samples (burn of 1/2)
projection_dim = 500 #random projection dimension
Ms = np.unique(np.logspace(0, 3, 10, dtype=int))
anms = ['GIGA', 'FW', 'RND']
n_trials = 50


for dnm in dnames:
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data(fldr+'/'+dnm+'.npz')

  print('Computing Laplace approximation')
  t0 = time.time()
  res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0)[:D], jac=lambda mu : -grad_log_likelihood(Z, mu).sum(axis=0) - grad_log_prior(mu))
  mu = res.x
  cov = -np.linalg.inv(hess_log_joint(Z, mu))
  t_laplace = time.time() - t0

  cputs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  csizes = np.zeros((len(anms), n_trials, Ms.shape[0]))
  Fs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  Fs_full = np.zeros(n_trials)
  cputs_full = np.zeros(n_trials)

  for tr in range(n_trials):
    print('Trial ' + str(tr+1) +'/' + str(n_trials))

    print('Computing random projection')
    t0 = time.time()
    proj = bc.ProjectionF(Z, grad_log_likelihood, projection_dim, lambda : np.random.multivariate_normal(mu, cov)) 
    vecs = proj.get()
    t_projection = time.time()-t0

    print('Running MCMC on the full dataset')
    accept_rate = 1.
    mcmc_attempt = 1
    while (accept_rate < .15 or accept_rate > 0.7):
      t0 = time.time()
      mh_param_init = np.random.multivariate_normal(mu, cov)
      th_samples, accept_rate = mh(mh_param_init,
             lambda th : log_joint(Z, th, np.ones(Z.shape[0])),
             None,
             lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(D)),
             steps=mh_steps,
             thin=mh_thin,
             target_rate=mh_target,
             proposal_param=mh_step_var_init)
      cputs_full[tr] = time.time()-t0
      print('attempt ' + str(mcmc_attempt) + ': accept rate = ' + str(accept_rate) + ', passes if in (.15, .7) ')
      mcmc_attempt += 1
    th_samples = np.array(th_samples)
    Fs_full[tr] = 0. #always 0, just doing this to make later code simpler
    full_samples = np.array(th_samples)
    
    
    print('Running coreset construction / MCMC')
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
      
        print('M = ' + str(Ms[m]) + ': metropolis hastings')
        accept_rate = 1.
        mcmc_attempt = 1
        while (accept_rate < .15 or accept_rate > 0.7):
          t0 = time.time()
          mh_param_init = np.random.multivariate_normal(mu, cov)
          th_samples, accept_rate = mh(mh_param_init,
                 lambda th : log_joint(Z[idcs, :], th, wts[idcs]),
                 None,
                 lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(D)),
                 steps=mh_steps,
                 thin=mh_thin,
                 target_rate=mh_target,
                 proposal_param=mh_step_var_init)
          t_alg_mh = time.time()-t0    
          print('attempt ' + str(mcmc_attempt) + ': accept rate = ' + str(accept_rate) + ', passes if in (.15, .7) ')
          mcmc_attempt += 1
        th_samples = np.array(th_samples)
        print('M = ' + str(Ms[m]) + ': CPU times')
        cputs[aidx, tr, m] = t_laplace + t_projection + t_setup + t_alg + t_alg_mh
        print('M = ' + str(Ms[m]) + ': coreset sizes')
        csizes[aidx, tr, m] = (wts>0).sum()
        print('M = ' + str(Ms[m]) + ': F norms')
        gcs = np.array([ (grad_log_likelihood(Z[idcs, :], full_samples[i, :])*wts[idcs][:, np.newaxis]).sum(axis=0) for i in range(full_samples.shape[0]) ])
        gfs = np.array([ (grad_log_likelihood(Z, full_samples[i, :])).sum(axis=0) for i in range(full_samples.shape[0]) ])
        Fs[aidx, tr, m] = (((gcs - gfs)**2).sum(axis=1)).mean()

  np.savez_compressed(fldr+'/'+dnm+'_results.npz', Ms=Ms, Fs=Fs, Fs_full=Fs_full, cputs=cputs, cputs_full=cputs_full, csizes=csizes, anms=anms)
        

