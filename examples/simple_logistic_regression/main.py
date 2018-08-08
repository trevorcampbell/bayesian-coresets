from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
from inference import nuts, rhat, hmc
import time

def gen_synthetic(n):
  mu = np.array([0, 0])
  cov = np.eye(2)
  th = np.array([3, 3])
  X = np.random.multivariate_normal(mu, cov, n)
  ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
  y =(np.random.rand(n) <= ps).astype(int)
  return y[:, np.newaxis]*X, (y[:, np.newaxis]*X).mean(axis=0)

def log_joint(Z, th, wts):
  return (wts*log_likelihood(Z, th)).sum() + log_prior(th)

def log_likelihood(z, th):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = -np.log1p(np.exp(m))
    else:
      m = -m
    return m 
    #-np.log1p(np.exp(-(th*z).sum()))
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = -np.log1p(np.exp(m[idcs]))
    m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
    return m
    #return -np.log1p(np.exp(-(th*z).sum(axis=1)))

def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

def grad_log_likelihood(z, th, idx=None):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = np.exp(m)/(1.+np.exp(m))
    else:
      m = 1.
    return m*z
    #es = np.exp(-(th*z).sum())
    #return es/(1.+es)*z
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
    m[np.logical_not(idcs)] = 1.
    if idx is None:
      return m[:, np.newaxis]*z
    return m*z[:, idx]
    #es = np.exp(-(th*z).sum(axis=1))
    #return (es/(1.+es))[:, np.newaxis]*z

def grad_log_prior(th):
  return -th

def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

def hess_log_joint(z, th):
  #m = -(th*z).sum(axis=1)
  #idcs = m < 100
  #m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
  #m[np.logical_not(idcs)] = 1.
  #H_log_like = -(z.T).dot((m**2)[:, np.newaxis]*z)
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior


mcmc_steps = 5000 #total number of MH steps
mcmc_burn = 1000
projection_dim = 500 #random projection dimension
Ms = np.unique(np.logspace(0, 3, 10, dtype=int))
pbar = True #progress bar display flag
step_size_init = 0.001
n_leap = 15
target_a = 0.8
anms = ['GIGA', 'FW', 'RND']
sampler = hmc #nuts

#generate the synthetic dataset
mu = np.array([0, 0])
cov = np.eye(2)
th = np.array([3, 3])
X = np.random.multivariate_normal(mu, cov, n)
ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
y =(np.random.rand(n) <= ps).astype(int)
Z = y[:, np.newaxis]*X
Zmean = Z.mean(axis=0)

#run MCMC on the full dataset
logpZ = lambda th : log_joint(Z, th, np.ones(Z.shape[0]))
glogpZ = lambda th : grad_log_joint(Z, th, np.ones(Z.shape[0]))
mcmc_param_init = np.random.multivariate_normal(mu, cov)
t0 = time.time()
full_samples = sampler(logp = logpZ, gradlogp = glogpZ, 
                 x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, 
                 n_leapfrogs = n_leap, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a) 
cputs_full[tr] = time.time()-t0
chains[tr, :, :] = full_samples


#find a cheap posterior approx using laplace approx
res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0)[:D], jac=lambda mu : -grad_log_joint(Z, mu, np.ones(Z.shape[0])))
cov = -np.linalg.inv(hess_log_joint(Z, mu))
var_scales=np.ones(cov.shape[0])

#build the projection
proj = bc.ProjectionF(Z, grad_log_likelihood, projection_dim, lambda : np.random.multivariate_normal(mu, cov)) 
vecs = proj.get()

#build the coreset
giga = bc.GIGA(vecs)
giga.run(M)
wts = giga.weights()
idcs = wts > 0

#run MCMC on the coreset
logpZ = lambda th : log_joint(Z[idcs, :], th, wts[idcs])
glogpZ = lambda th : grad_log_joint(Z[idcs, :], th, wts[idcs])
mcmc_param_init = np.random.multivariate_normal(mu, cov)
print('M = ' + str(Ms[m]) + ': MCMC')
t0 = time.time()
th_samples = sampler(logp=logpZ, gradlogp=glogpZ, 
             x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, 
             n_leapfrogs= n_leap, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a)
t_alg_mcmc = time.time()-t0    

 


