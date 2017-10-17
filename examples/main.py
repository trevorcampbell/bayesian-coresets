import numpy as np
from algs import *
import hilbertcoresets as hc
from scipy.optimize import minimize

## FOR LOGISTIC REGRESSION
from model_lr import *
## FOR POISSON REGRESSION
#from model_poiss import *

n = 10000 #amt of data to generate
mh_steps = 100000 #total number of MH steps
mh_thin = 5 #thinning factor
mh_target = 0.234 #target acceptance rate
mh_step_var_init = 0.1 #initial step variance
n_samples = mh_steps / 2 / mh_thin #number of output samples (burn of 1/2)
projection_dim = 500 #random projection dimension
Ms = [10, 50, 100, 500, 1000, 5000, 10000] #values of M to sweep over

print 'Generating synthetic data'
Z, th0 = gen_synthetic(n)

print 'Computing Laplace approximation'
res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), th0, jac=lambda mu : -grad_log_likelihood(Z, mu).sum(axis=0) - grad_log_prior(mu))
mu_laplace = res.x
cov_laplace = -np.linalg.inv(hess_log_joint(Z, mu_laplace))

print 'Setting up preproc algs (FW, IS, Random)'
alg_names = ['FW-F', 'IS-F', 'Uniform', 'Random']
algs = []
algs.append(hc.ProjectedFrankWolfe(Z, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, lambda : np.random.multivariate_normal(mu_laplace, cov_laplace)))
algs.append(hc.ProjectedImportanceSampling(Z, log_likelihood, log_prior, grad_log_likelihood, grad_log_prior, hess_log_joint, projection_dim, lambda: np.random.multivariate_normal(mu_laplace, cov_laplace)))
algs.append(hc.RandomSubsample(Z.shape[0]))

print 'Running MCMC on the full dataset for comparison'
full_alg = hc.FullDataset(Z.shape[0])
mh_param_init = np.random.multivariate_normal(np.zeros(2), np.eye(2))
th_samples, accept_rate = mh(mh_param_init,
       lambda th : log_joint(Z, th,np.ones(Z.shape[0])),
       None,
       lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(2)),
       steps=mh_steps,
       thin=mh_thin,
       target_rate=mh_target,
       proposal_param=mh_step_var_init)
full_samples = np.array(th_samples)

print 'Running coreset construction / MCMC'
for alg, anm in zip(algs, alg_names):
  print anm +':'
  for m, M in enumerate(Ms):
    print 'M = ' + str(M) + ': coreset construction'
    #this runs alg up to a level of M; on the next iteration, it will continue from where it left off
    alg.run(M)
    wts = alg.wts.copy()
    idcs = wts > 0
  
    print 'M = ' + str(M) + ': metropolis hastings'
    mh_param_init = np.random.multivariate_normal(np.zeros(2), np.eye(2))
    th_samples, accept_rate = mh(mh_param_init,
           lambda th : log_joint(Z[idcs, :], th, wts[idcs]),
           None,
           lambda th, sig : np.random.multivariate_normal(th, sig*np.eye(2)),
           steps=mh_steps,
           thin=mh_thin,
           target_rate=mh_target,
           proposal_param=mh_step_var_init)
    print 'M = ' + str(M) + ': computing 1-wasserstein'
    w1 = wasserstein1(np.array(th_samples), full_samples)
    print 'W1 = ' + str(wasserstein1(np.array(th_samples), full_samples))
