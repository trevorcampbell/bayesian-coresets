from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize


#######################################
#######################################
## Step 0: Generate a Synthetic Dataset
#######################################
#######################################

N = 10000
D = 10

mu = np.zeros(D)
cov = np.eye(D)
th = 3.*np.ones(D)
X = np.random.multivariate_normal(mu, cov, N)
ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
y =(np.random.rand(N) <= ps).astype(int)
Z = y[:, np.newaxis]*X
Zmean = Z.mean(axis=0)


###########################
###########################
## Step 1: Define the Model
###########################
###########################

from model import *

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
M = 100
giga = bc.GIGA(vecs)
giga.run(M)
wts = giga.weights()
idcs = wts > 0

########################
########################
## Step 5: Run Inference
########################
########################

#example:
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
