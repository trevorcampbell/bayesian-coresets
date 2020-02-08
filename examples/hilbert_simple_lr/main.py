from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
import sys, os
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import gaussian


np.random.seed(1)

#######################################
#######################################
## Step 0: Generate a Synthetic Dataset
#######################################
#######################################

print('Generating data...')

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
## Step 1: Define the model
###########################
###########################

print('Importing model functions...')

from model_lr import *

###############################################################
###############################################################
## Step 2: Pick a location for the coreset tangent space
###############################################################
###############################################################

print('Finding MAP for tangent space approximation...')

#Here we use the laplace approximation of the posterior
#first, optimize the log joint to find the mode:
res = minimize(lambda mu : -log_joint(Z, mu, np.ones(Z.shape[0]))[0], Z.mean(axis=0), jac=lambda mu : -grad_th_log_joint(Z, mu, np.ones(Z.shape[0]))[0,:])
#then find a quadratic expansion around the mode, and assume the distribution is Gaussian
mu = res.x
cov = -np.linalg.inv(hess_th_log_joint(Z, mu, np.ones(Z.shape[0]))[0, :, :])

#you can replace this step with a lot of different things: e.g.
# - choose a subset of data uniformly and weight uniformly, run MCMC
# - same thing with var inf, INLA, SGLD, etc 

##########################################################################
##########################################################################
## Step 3: Compute a random finite projection of the tangent space  
##########################################################################
##########################################################################

projection_dim = 500 #random projection dimension
sampler = lambda sz, w, p : np.atleast_2d(np.random.multivariate_normal(mu, cov, sz))
projector = bc.BlackBoxProjector(sampler, projection_dim, log_likelihood)

############################
############################
## Step 4: Build the Coreset
############################
############################


print('Building the coreset...')

#build the coreset
M = 500 # use up to 500 datapoints (run 500 itrs)
coreset = bc.HilbertCoreset(Z, projector) #do coreset construction using the discretized log-likelihood functions
coreset.build(M, M) #build the coreset to size M with at most M iterations
wts, pts, idcs = coreset.get() #get the output weights
print('coreset weights:')
print(wts)
print('coreset pts:')
print(pts)
print('coreset idcs:')
print(idcs)


##############################
##############################
## Step 5: Evaluate coreset
##############################
##############################

#Normally at this point we'd run posterior inference on the coreset
#But for this (illustrative) example we will evaluate quality via Laplace posterior approx

print('Evaluating coreset quality...')

w = np.zeros(N)
w[idcs] = wts

#compute error using laplace approx
res = minimize(lambda mu : -log_joint(Z, mu, w)[0], Z.mean(axis=0), jac=lambda mu : -grad_th_log_joint(Z, mu, w)[0,:])
muw = res.x
#then find a quadratic expansion around the mode, and assume the distribution is Gaussian
covw = -np.linalg.inv(hess_th_log_joint(Z, muw, w)[0,:,:])

print('Done!') 

#compare posterior and coreset
np.set_printoptions(linewidth=10000)
print('Posterior requires ' + str(N) + ' data')
print('mu, cov = ' + str(mu) + '\n' + str(cov))
print('Coreset requires ' + str(idcs.shape[0]) + ' data')
print('muw, covw = ' + str(muw) + '\n' + str(covw))

print('KL(coreset || posterior) = ' + str(gaussian.gaussian_KL(muw, covw, mu, np.linalg.inv(cov))))

