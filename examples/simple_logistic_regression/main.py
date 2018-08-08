from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize


#######################################
#######################################
## Step 0: Generate a Synthetic Dataset
#######################################
#######################################

mu = np.array([0, 0])
cov = np.eye(2)
th = np.array([3, 3])
X = np.random.multivariate_normal(mu, cov, n)
ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
y =(np.random.rand(n) <= ps).astype(int)
Z = y[:, np.newaxis]*X
Zmean = Z.mean(axis=0)


###########################
###########################
## Step 1: Define the Model
###########################
###########################

#computes the logistic regression log-likelihood for data z and parameter th
#input: z = N x D numpy array, th = length D numpy array
#output: length N numpy array of log_likelihoods
def log_likelihood(z, th):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = -np.log1p(np.exp(m))
    else:
      m = -m
    return m 
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = -np.log1p(np.exp(m[idcs]))
    m[np.logical_not(idcs)] = -m[np.logical_not(idcs)]
    return m

#computes the gradient of the logistic regression log-likelihood
# z is data, one row per vector; th is the parameter
#input: z = N x D numpy array, th = length D numpy array, idx = optional gradient component index
#output: (if idx = None): N x D array of gradients  (if idx = integer) N x 1 array of gradient components
def grad_log_likelihood(z, th, idx=None):
  if len(z.shape) == 1:
    m = -(th*z).sum()
    if m < 100:
      m = np.exp(m)/(1.+np.exp(m))
    else:
      m = 1.
    return m*z
  else:
    m = -(th*z).sum(axis=1)
    idcs = m < 100
    m[idcs] = np.exp(m[idcs])/(1.+np.exp(m[idcs]))
    m[np.logical_not(idcs)] = 1.
    if idx is None:
      return m[:, np.newaxis]*z
    return m*z[:, idx]

#computes the log prior for parameter th
#input: th = length D numpy array
#output: log prior density value, scalar
def log_prior(th):
  return -0.5*th.shape[0]*np.log(2.*np.pi) - 0.5*(th**2).sum()

#computes the log prior gradient for parameter th
#input: th = length D numpy array
#output: length D numpy gradient array
def grad_log_prior(th):
  return -th

#computes the log joint probability for data z and parameter th, where the data are weighted by wts
#input: Z = N x D numpy array, th = length D numpy array, wts = length N numpy array of nonnegative values
#output: weighted log joint, scalar
def log_joint(Z, th, wts):
  return (wts*log_likelihood(Z, th)).sum() + log_prior(th)

#same as above; outputs length D numpy array gradient
def grad_log_joint(z, th, wts):
  return grad_log_prior(th) + (wts[:, np.newaxis]*grad_log_likelihood(z, th)).sum(axis=0)

#same as above; outputs D x D numpy array Hessian
def hess_log_joint(z, th):
  es = np.exp(-(th*z).sum(axis=1))
  H_log_like = -(z.T).dot((es/(1.+es)**2)[:, np.newaxis]*z)
  H_log_prior = -np.eye(th.shape[0])
  return H_log_like + H_log_prior


#################################################
#################################################
## Step 2: Obtain a Cheap Posterior Approximation
#################################################
#################################################

res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0)[:D], jac=lambda mu : -grad_log_joint(Z, mu, np.ones(Z.shape[0])))
cov = -np.linalg.inv(hess_log_joint(Z, mu))
var_scales=np.ones(cov.shape[0])


########################################
########################################
## Step 3: Discretize the Log Likelihood
########################################
########################################

projection_dim = 500 #random projection dimension
proj = bc.ProjectionF(Z, grad_log_likelihood, projection_dim, lambda : np.random.multivariate_normal(mu, cov)) 
vecs = proj.get()

############################
############################
## Step 4: Build the Coreset
############################
############################


#build the coreset
giga = bc.GIGA(vecs)
giga.run(M)
wts = giga.weights()
idcs = wts > 0

########################
########################
## Step 5: Run Inference
########################
########################

##for example:
#from inference import hmc
#mcmc_steps = 5000 #total number of MH steps
#mcmc_burn = 1000
#step_size_init = 0.001
#n_leap = 15
#target_a = 0.8
#pbar = True #progress bar display flag
#logpZ = lambda th : log_joint(Z[idcs, :], th, wts[idcs])
#glogpZ = lambda th : grad_log_joint(Z[idcs, :], th, wts[idcs])
#mcmc_param_init = np.random.multivariate_normal(mu, cov)
#th_samples = hmc(logp=logpZ, gradlogp=glogpZ, 
#             x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, 
#             n_leapfrogs= n_leap, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a)
