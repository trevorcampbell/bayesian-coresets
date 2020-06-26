from __future__ import print_function
import numpy as np
import scipy.linalg as sl
import pickle as pk 
import os, sys
import argparse
import bayesiancoresets as bc
from scipy.optimize import minimize, nnls
import time

#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from mcmc import sampler
import model_gaussian as model

#computes the Laplace approximation N(mu, Sig) to the posterior with weights wts
def get_laplace(wts, Z, mu0, diag = False):
  trials = 10
  Zw = Z[wts>0, :]
  ww = wts[wts>0]
  while True:
    try:
      res = minimize(lambda mu : -log_joint(Zw, mu, ww)[0], mu0, jac=lambda mu : -grad_th_log_joint(Zw, mu, ww)[0,:])
    except:
      mu0 = mu0.copy()
      mu0 += np.sqrt((mu0**2).sum())*0.1*np.random.randn(mu0.shape[0])
      trials -= 1
      if trials <= 0:
        print('Tried laplace opt 10 times, failed')
        break
      continue
    break
  mu = res.x
  if diag:
    LSigInv = np.sqrt(-diag_hess_th_log_joint(Zw, mu, ww)[0,:])
    LSig = 1./LSigInv
  else:
    LSigInv = np.linalg.cholesky(-hess_th_log_joint(Zw, mu, ww)[0,:,:])
    LSig = sl.solve_triangular(LSigInv, np.eye(LSigInv.shape[0]), lower=True, overwrite_b = True, check_finite = False)
  return mu, LSig, LSigInv

#######################################
#######################################
## Step 0: Parse Arguments
#######################################
#######################################

parser = argparse.ArgumentParser(description="Runs Riemannian logistic or poisson regression (employing coreset contruction) on the specified dataset")
parser.add_argument('dnm', type=str, help="the name of the dataset on which to run regression")
parser.add_argument('model', type=str, choices=["lr","poiss"], help="The regression model to use. lr refers to logistic regression, and poiss refers to poisson regression.")
parser.add_argument('alg', type=str, help="The algorithm to use for regression - should be one of GIGAO / GIGAR / RAND / PRIOR / SVI") #TODO: find way to make this help message autoupdate with new methods
parser.add_argument('ID', type=int, help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--fldr', type=str, default="results/", help="This script will save results in this folder. Default \"results/\"")
parser.add_argument('--use_diag_laplace_w', action='store_const', default = False, const=True, help="")
parser.add_argument('--M', type=int, default=20, help="Maximum allowable coreset size")
parser.add_argument('--SVI_opt_itrs', type=int, default = '1500', help = '(If using SVI/HOPS) The number of iterations used when optimizing weights.')
parser.add_argument('--SVI_step_sched', type=str, default = "lambda i : 1./(1+i)", help="Step schedule (tuning rate) for SVI & HOPS, entered as a lambda expression in quotation marks. Default is \"lambda i : 1./(1+i)\"")
parser.add_argument('--n_subsample_opt', type=int, default=400, help="(If using Sparse VI/HOPS) the size of the random subsample to use when optimizing the coreset weights in each reweight step")
parser.add_argument('--n_susample_select', type=int, default=1000, help="(If using Sparse VI/HOPS) the size of the random subsample to use when determining which point to add to the coreset in each select step")
parser.add_argument('--proj_dim', type=int, default = '100', help = "The number of samples to take when discretizing log likelihoods")
parser.add_argument('--pihat_noise', type=float, default=.75, help = "(If calling GIGAR or simulating another realistically tuned Hilbert Coreset) - a measure of how much noise to introduce to the smoothed pi-hat to make the sampler")
parser.add_argument("--mcmc_samples", type=int, default=10000, help="number of MCMC samples to take (we also take this many warmup steps before sampling)")

arguments = parser.parse_args()
model = arguments.model
dnm = arguments.dnm
alg = arguments.alg
ID = arguments.ID
fldr = arguments.fldr
N_samples = arguments.mcmc_samples

###############################
## TUNING PARAMETERS ##
use_diag_laplace_w = arguments.use_diag_laplace_w
M = arguments.M
SVI_step_sched = eval(arguments.SVI_step_sched)
n_subsample_opt = arguments.n_subsample_opt
n_subsample_select = arguments.n_subsample_select
projection_dim = arguments.proj_dim
pihat_noise = arguments.pihat_noise
SVI_opt_itrs = arguments.SVI_opt_itrs
###############################


print('running ' + str(dnm)+ ' ' + str(alg)+ ' ' + str(ID))

#######################################
#######################################
## Step 1: Load Dataset
#######################################
#######################################

print('Loading dataset '+dnm)
data_X, data_Y, Z, Zt, D = load_data('../data/'+dnm+'.npz')

###########################
###########################
## Step 2: Define the model
###########################
###########################

if model=='lr':
  from model_lr import *
else:
  from model_poiss import *

##########################################################################
##########################################################################
## Step 3: Compute a random finite projection of the tangent space  
##########################################################################
##########################################################################

np.random.seed(ID)

#run sampler
sample_caching_folder = "caching/"
samples = sampler(dnm, data_X, data_Y, N_samples, stan_representation, sample_caching_folder = sample_caching_folder)
#TODO FIX SAMPLER TO NOT HAVE TO DO THIS
samples = np.hstack((samples[:, 1:], samples[:, 0][:,np.newaxis]))

#fit a gaussian to the posterior samples 
#used for pihat computation for Hilbert coresets with noise to simulate uncertainty in a good pihat
mup = samples.mean(axis=0)
Sigp = np.cov(samples, rowvar=False)
LSigp = np.linalg.cholesky(Sigp)
LSigpInv = sl.solve_triangular(LSigp, np.eye(LSigp.shape[0]), lower=True, overwrite_b = True, check_finite = False)

#create the prior -- also used for the above purpose
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

#get pihat via interpolation between prior/posterior + noise
#uniformly smooth between prior and posterior
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2.*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)

print('Building projectors')
sampler_optimal = lambda sz, w, pts : mup + np.random.randn(sz, mup.shape[0]).dot(LSigp.T)
sampler_realistic = lambda sz, w, pts : muhat + np.random.randn(sz, muhat.shape[0]).dot(LSighat.T)
def sampler_w(sz, w, pts):
  if pts.shape[0] == 0:
    w = np.zeros(1)
    pts = np.zeros((1, Z.shape[1]))
  muw, LSigw, LSigwInv = get_laplace(w, pts, mu0, use_diag_laplace_w)
  if use_diag_laplace_w:
    return muw + np.random.randn(sz, muw.shape[0])*LSigw
  else:
    return muw + np.random.randn(sz, muw.shape[0]).dot(LSigw.T)

prj_optimal = bc.BlackBoxProjector(sampler_optimal, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, projection_dim, log_likelihood, grad_z_log_likelihood)
prj_w = bc.BlackBoxProjector(sampler_w, projection_dim, log_likelihood, grad_z_log_likelihood)

############################
############################
## Step 4: Build the Coreset
############################
############################
 
print('Creating coresets object')
#create coreset construction objects
t0 = time.perf_counter()
giga_optimal = bc.HilbertCoreset(Z, prj_optimal)
gigao_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
giga_realistic = bc.HilbertCoreset(Z, prj_realistic)
gigar_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
unif = bc.UniformSamplingCoreset(Z)
unif_t_setup = time.perf_counter()-t0

t0 = time.perf_counter()
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs=SVI_opt_itrs, n_subsample_opt = n_subsample_opt, n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
sparsevi_t_setup = time.perf_counter()-t0

#t0 = time.perf_counter()

algs = {'SVI': sparsevi,
        'GIGAO': giga_optimal, 
        'GIGAR': giga_realistic, 
        'RAND': unif,
        'PRIOR': None}
coreset = algs[alg]
t0s = {'SVI' : sparsevi_t_setup,
       'GIGAO' : gigao_t_setup,
       'GIGAR' : gigar_t_setup,
       'RAND' : unif_t_setup,
       'PRIOR' : 0.}

print('Building coresets via ' + alg)
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]
cputs = np.zeros(M+1)
cputs[0] = t0s[alg]
for m in range(1, M+1):
  print(str(m)+'/'+str(M))
  if alg != 'PRIOR':
    t0 = time.perf_counter()
    coreset.build(1)
    cputs[m] = time.perf_counter()-t0

    #record time and weights
    wts, pts, idcs = coreset.get()
    w.append(wts)
    p.append(pts)
  else:
    w.append(np.array([0.]))
    p.append(np.zeros((1, Z.shape[1])))

##############################
##############################
## Step 5: Evaluate coreset
##############################
##############################
    
#get laplace approximations for each weight setting, and KL divergence to full posterior laplace approx mup Sigp
#used for a quick/dirty performance comparison without expensive posterior sample comparisons (e.g. energy distance)
mus_laplace = np.zeros((M+1, D))
Sigs_laplace = np.zeros((M+1, D, D))
rkls_laplace = np.zeros(M+1)
fkls_laplace = np.zeros(M+1)
print('Computing coreset Laplace approximation + approximate KL(posterior || coreset laplace)')
for m in range(M+1):
  mul, LSigl, LSiglInv = get_laplace(w[m], p[m], Z.mean(axis=0)[:D])
  mus_laplace[m,:] = mul
  Sigs_laplace[m,:,:] = LSigl.dot(LSigl.T)
  rkls_laplace[m] = model.gaussian_KL(mul, Sigs_laplace[m,:,:], mup, LSigpInv.dot(LSigpInv.T))
  fkls_laplace[m] = model.gaussian_KL(mup, Sigp, mul, LSiglInv.dot(LSiglInv.T))

#save results
if not os.path.exists(fldr):
  os.mkdir(fldr)

#make hash of step schedule so it can be encoded in the file name:
SVI_step_sched_hash_sha1 = hashlib.sha1(arguments.SVI_step_sched.encode('utf-8')).hexdigest()

f = open(os.path.join(fldr, dnm+'_'+model+'_'+alg+'_results_'+'id='+str(ID)+'_mcmc_samples='+str(N_samples)+'_use_diag_laplace_w='+str(use_diag_laplace_w)+'_proj_dim='+str(projection_dim)+'_SVI_opt_itrs='+str(SVI_opt_itrs)+'_n_subsample_opt='+str(n_subsample_opt)+'_n_subsample_select='+str(n_subsample_select)+'_'+'SVI_step_sched_hash_sha1='+SVI_step_sched_hash_sha1+'_pihat_noise='+str(pihat_noise)+'.pk', 'wb')
res = (cputs, w, p, mus_laplace, Sigs_laplace, rkls_laplace, fkls_laplace, dnm, model, alg, ID, N_samples, use_diag_laplace_w, projection_dim, SVI_opt_itrs, n_subsample_opt, n_subsample_select, SVI_step_sched, pihat_noise)
pk.dump(res, f)
f.close()