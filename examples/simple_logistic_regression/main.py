from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize


#######################################
#######################################
## Step 0: Generate a Synthetic Dataset
#######################################
#######################################


#10,000 datapoints, 10-dimensional
N = 10000
D = 10
#generate input vectors from standard normal
mu = np.zeros(D)
cov = np.eye(D)
X = np.random.multivariate_normal(mu, cov, N)
#set the true parameter to [3,3,3,..3]
th = 3.*np.ones(D)
#generate responses given inputs
ps = 1.0/(1.0+np.exp(-(X*th).sum(axis=1)))
y =(np.random.rand(N) <= ps).astype(int)
y[y==0] = -1
#format data for (grad/hess) log (likelihood/prior/joint)
Z = y[:, np.newaxis]*X

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

#Here we use the laplace approximation
#first, optimize the log joint to find the mode:
res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0])), Z.mean(axis=0), jac=lambda mu : -grad_log_joint(Z, mu, np.ones(Z.shape[0])))
#then find a quadratic expansion around the mode, and assume the distribution is Gaussian
cov = -np.linalg.inv(hess_log_joint(Z, mu))

#we can call post_approx() to sample from the approximate posterior
post_approx = lambda : np.random.multivariate_normal(mu, cov)

#you can replace this step with almost any inference alg: subset MCMC, variational inference, INLA, SGLD, etc

########################################
########################################
## Step 3: Discretize the Log Likelihood
########################################
########################################

projection_dim = 500 #random projection dimension, K
#build the discretization of all the log-likelihoods based on random projection
proj = bc.ProjectionF(Z, grad_log_likelihood, projection_dim, post_approx) 
#construct the N x K discretized log-likelihood matrix; each row represents the discretized LL func for one datapoint
vecs = proj.get()

############################
############################
## Step 4: Build the Coreset
############################
############################

#build the coreset
M = 100 # use 100 datapoints
giga = bc.GIGA(vecs) #do coreset construction using the discretized log-likelihood functions
giga.run(M) #build the coreset
wts = giga.weights() #get the output weights
idcs = wts > 0 #pull out the indices of datapoints that were included in the coreset

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
#var_scales=np.ones(cov.shape[0])
#pbar = True #progress bar display flag
#logpZ = lambda th : log_joint(Z[idcs, :], th, wts[idcs])
#glogpZ = lambda th : grad_log_joint(Z[idcs, :], th, wts[idcs])
#mcmc_param_init = np.random.multivariate_normal(mu, cov)
#th_samples = hmc(logp=logpZ, gradlogp=glogpZ, 
#             x0 = mcmc_param_init, sample_steps=mcmc_steps, burn_steps=mcmc_burn, adapt_steps=mcmc_burn, 
#             n_leapfrogs= n_leap, scale=var_scales, progress_bar=pbar, step_size=step_size_init, target_accept=target_a)
