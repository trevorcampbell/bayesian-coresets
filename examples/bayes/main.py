import numpy as np
from inference import mh, wasserstein1
import hilbertcoresets as hc
from scipy.optimize import minimize


## FOR LOGISTIC REGRESSION
from model_lr import *
dnames = ['synth', 'ds1', 'phishing']
folder = 'lr'

### FOR POISSON REGRESSION
#from model_poiss import *
#dnames = ['synth', 'airportdelays', 'biketrips']
#folder = 'poiss'


#mh_steps = 100000 #total number of MH steps
#mh_thin = 5 #thinning factor
#mh_target = 0.234 #target acceptance rate
#mh_step_var_init = 0.1 #initial step variance
#n_samples = mh_steps / 2 / mh_thin #number of output samples (burn of 1/2)
#projection_dim = 500 #random projection dimension
#Ms = np.array([10, 50, 100, 500, 1000, 5000, 10000]) #values of M to sweep over
#anms = ['GIGA', 'FW', 'RND']
#n_trials = 10


#Fast test params
mh_steps = 1000 #total number of MH steps
mh_thin = 5 #thinning factor
mh_target = 0.234 #target acceptance rate
mh_step_var_init = 0.1 #initial step variance
n_samples = mh_steps / 2 / mh_thin #number of output samples (burn of 1/2)
projection_dim = 200 #random projection dimension
Ms = np.array([10, 100, 1000]) #values of M to sweep over
anms = ['GIGA', 'FW', 'RND']
n_trials = 2



for dnm in dnames:
  print 'Loading dataset '+dnm
  Z, Zt = load_data(folder+'/'+dnm+'.npz')

  print 'Computing Laplace approximation'
  res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0), jac=lambda mu : -grad_log_likelihood(Z, mu).sum(axis=0) - grad_log_prior(mu))
  mu = res.x
  cov = -np.linalg.inv(hess_log_joint(Z, mu))

  print 'Computing Max TestLL'
  res = minimize(lambda mu : -log_joint(Zt, mu, np.ones(Zt.shape[0])), Zt.mean(axis=0), jac=lambda mu : -grad_log_likelihood(Zt, mu).sum(axis=0) - grad_log_prior(mu))
  ll_max = res.fun/Zt.shape[0]

  w1s = np.zeros((len(anms), n_trials, Ms))
  lls = np.zeros((len(anms), n_trials, Ms))
  cputs = np.zeros((len(anms), n_trials, Ms))
  ll_fulls = np.zeros(n_trials)

  for tr in range(n_trials):
    print 'Trial ' + str(tr+1) +'/' + str(n_trials)

    print 'Computing random F projection'
    proj = hc.ProjectionF(Z, grad_log_likelihood, projection_dim, lambda : np.random.multivariate_normal(mu, cov)) 
    vecs = proj.get()

    print 'Running MCMC on the full dataset'
    full_alg = hc.FullDataset(vecs)
    D = Z.shape[1]
    mh_param_init = np.random.multivariate_normal(np.zeros(D), np.eye(D))
    th_samples, accept_rate = mh(mh_param_init,
           lambda th : log_joint(Z, th, np.ones(Z.shape[0])),
           None,
           lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(2)),
           steps=mh_steps,
           thin=mh_thin,
           target_rate=mh_target,
           proposal_param=mh_step_var_init)
    full_samples = np.array(th_samples)
    ll_full = np.array([ log_likelihood(Zt, full_samples[i, :]).sum()/Zt.shape[0] for i in range(full_samples.shape[0]) ]).sum()/full_samples.shape[0]
    ll_fulls[tr] = ll_full
    
    print 'Running coreset construction / MCMC'
    for aidx, anm in enumerate(anms):
      print anm +':'

      alg = None
      if 'GIGA' in anm:
        alg = hc.GIGA(vecs)
      elif anm == 'FW':
        alg = hc.FrankWolfe(vecs)
      else:
        alg = hc.RandomSubsampling(vecs) 

      for m in range(Ms.shape[0]):
        print 'M = ' + str(M) + ': coreset construction'
        #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
        alg.run(Ms[m])
        wts = alg.weights()
        idcs = wts > 0
      
        print 'M = ' + str(Ms[m]) + ': metropolis hastings'
        mh_param_init = np.random.multivariate_normal(np.zeros(2), np.eye(2))
        th_samples, accept_rate = mh(mh_param_init,
               lambda th : log_joint(Z[idcs, :], th, wts[idcs]),
               None,
               lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(2)),
               steps=mh_steps,
               thin=mh_thin,
               target_rate=mh_target,
               proposal_param=mh_step_var_init)
        print 'M = ' + str(Ms[m]) + ': computing 1-wasserstein'
        w1 = wasserstein1(np.array(th_samples), full_samples)
        ll = np.array([ log_likelihood(Zt, th_samples[i, :]).sum()/Zt.shape[0] for i in range(th_samples.shape[0]) ]).sum()/th_samples.shape[0]
        print 'W1 = ' + str(w1)
        print 'LL = ' + str(ll)
        w1s[aidx, tr, m] = w1
        lls[aidx, tr, m] = ll
        cputs[aidx, tr, m] = AAA

  np.savez_compressed(folder+'/'+dnm+'_results.npz', w1s=w1s, lls=lls, cputs=cputs, ll_fulls=ll_fulls, ll_max=ll_max, anms=anms)
        

