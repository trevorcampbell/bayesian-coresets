import numpy as np
from inference import mh, wasserstein1, gaussian_KL
import hilbertcoresets as hc
from scipy.optimize import minimize
import time


# FOR LOGISTIC REGRESSION
from model_lr import *
dnames = ['synth', 'ds1', 'phishing']
fldr = 'lr'

#### FOR POISSON REGRESSION
#from model_poiss import *
#dnames = ['synth', 'airportdelays', 'biketrips']
#fldr = 'poiss'


mh_steps = 100000 #total number of MH steps
mh_thin = 5 #thinning factor
mh_target = 0.234 #target acceptance rate
mh_step_var_init = 0.1 #initial step variance
n_samples = mh_steps / 2 / mh_thin #number of output samples (burn of 1/2)
projection_dim = 500 #random projection dimension
Ms = np.array([10, 50, 100, 500, 1000, 5000, 10000]) #values of M to sweep over
anms = ['GIGA', 'FW', 'RND']
n_trials = 10


##Fast test params
#mh_steps = 1000 #total number of MH steps
#mh_thin = 5 #thinning factor
#mh_target = 0.234 #target acceptance rate
#mh_step_var_init = 0.1 #initial step variance
#n_samples = mh_steps / 2 / mh_thin #number of output samples (burn of 1/2)
#projection_dim = 100 #random projection dimension
#Ms = np.array([10, 50, 100, 500, 1000]) #values of M to sweep over
#anms = ['GIGA', 'FW', 'RND']
#n_trials = 3


for dnm in dnames:
  print 'Loading dataset '+dnm
  Z, Zt, D = load_data(fldr+'/'+dnm+'.npz')

  print 'Computing Laplace approximation'
  t0 = time.clock()
  res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0)[:D], jac=lambda mu : -grad_log_likelihood(Z, mu).sum(axis=0) - grad_log_prior(mu))
  mu = res.x
  cov = -np.linalg.inv(hess_log_joint(Z, mu))
  t_laplace = time.clock() - t0

  print 'Computing Max TestLL'
  res = minimize(lambda mu : -log_joint(Zt, mu, np.ones(Zt.shape[0])), Zt.mean(axis=0)[:D], jac=lambda mu : -grad_log_likelihood(Zt, mu).sum(axis=0) - grad_log_prior(mu))
  ll_max = res.fun/Zt.shape[0]

  w1s = np.zeros((len(anms), n_trials, Ms.shape[0]))
  lls = np.zeros((len(anms), n_trials, Ms.shape[0]))
  cputs = np.zeros((len(anms), n_trials, Ms.shape[0]))
  kls = np.zeros((len(anms), n_trials, Ms.shape[0]))
  lls_full = np.zeros(n_trials)
  cputs_full = np.zeros(n_trials)
  w1s_full = np.zeros(n_trials)
  kls_full = np.zeros(n_trials)

  print 'Running MCMC on the full dataset'
  mh_param_init = np.random.multivariate_normal(np.zeros(D), np.eye(D))
  th_samples, accept_rate = mh(mh_param_init,
         lambda th : log_joint(Z, th, np.ones(Z.shape[0])),
         None,
         lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(D)),
         steps=mh_steps,
         thin=mh_thin,
         target_rate=mh_target,
         proposal_param=mh_step_var_init)
  full_samples = np.array(th_samples)

  for tr in range(n_trials):
    print 'Trial ' + str(tr+1) +'/' + str(n_trials)

    print 'Computing random F projection'
    t0 = time.clock()
    proj = hc.ProjectionF(Z, grad_log_likelihood, projection_dim, lambda : np.random.multivariate_normal(mu, cov)) 
    vecs = proj.get()
    t_projection = time.clock()-t0

    print 'Running MCMC on the full dataset'
    t0 = time.clock()
    mh_param_init = np.random.multivariate_normal(np.zeros(D), np.eye(D))
    th_samples, accept_rate = mh(mh_param_init,
           lambda th : log_joint(Z, th, np.ones(Z.shape[0])),
           None,
           lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(D)),
           steps=mh_steps,
           thin=mh_thin,
           target_rate=mh_target,
           proposal_param=mh_step_var_init)
    cputs_full[tr] = time.clock()-t0
    th_samples = np.array(th_samples)
    lls_full[tr] = np.array([ log_likelihood(Zt, th_samples[i, :]).sum()/Zt.shape[0] for i in range(th_samples.shape[0]) ]).sum()/th_samples.shape[0]
    w1s_full[tr] = wasserstein1(th_samples, full_samples)
    kls_full[tr] = gaussian_KL(full_samples, th_samples)
    
    
    print 'Running coreset construction / MCMC'
    for aidx, anm in enumerate(anms):
      print anm +':'

      t0 = time.clock()
      alg = None
      if 'GIGA' in anm:
        alg = hc.GIGA(vecs)
      elif anm == 'FW':
        alg = hc.FrankWolfe(vecs)
      else:
        alg = hc.RandomSubsampling(vecs) 
      t_setup = time.clock() - t0

      t_alg = 0.
      for m in range(Ms.shape[0]):
        print 'M = ' + str(Ms[m]) + ': coreset construction'
        #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
        t0 = time.clock()
        alg.run(Ms[m])
        t_alg += time.clock()-t0
        wts = alg.weights()
        idcs = wts > 0
      
        print 'M = ' + str(Ms[m]) + ': metropolis hastings'
        t0 = time.clock()
        mh_param_init = np.random.multivariate_normal(np.zeros(D), np.eye(D))
        th_samples, accept_rate = mh(mh_param_init,
               lambda th : log_joint(Z[idcs, :], th, wts[idcs]),
               None,
               lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(D)),
               steps=mh_steps,
               thin=mh_thin,
               target_rate=mh_target,
               proposal_param=mh_step_var_init)
        t_alg_mh = time.clock()-t0    
        th_samples = np.array(th_samples)
        print 'M = ' + str(Ms[m]) + ': 1-wasserstein'
        w1s[aidx, tr, m] = wasserstein1(th_samples, full_samples)
        print 'M = ' + str(Ms[m]) + ': test log-likelihood'
        lls[aidx, tr, m] = np.array([ log_likelihood(Zt, th_samples[i, :]).sum()/Zt.shape[0] for i in range(th_samples.shape[0]) ]).sum()/th_samples.shape[0]
        print 'M = ' + str(Ms[m]) + ': Gaussian KL'
        kls[aidx, tr, m] = gaussian_KL(full_samples, th_samples)
        cputs[aidx, tr, m] = t_laplace + t_projection + t_setup + t_alg + t_alg_mh

  np.savez_compressed(fldr+'/'+dnm+'_results.npz', w1s=w1s, lls=lls, kls=kls, cputs=cputs, lls_full=lls_full, cputs_full=cputs_full, w1s_full=w1s_full, kls_full=kls_full, ll_max=ll_max, anms=anms)
        

