import numpy as np
import pickle as pk
import os, sys
hpc = True
if hpc:  sys.path.insert(1, os.path.join(sys.path[0], '/home/dm754/bayesian-coresets-private'))
import bayesiancoresets as bc
from scipy.stats import multivariate_normal
import time
import hashlib
import argparse
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import model_linreg

parser = argparse.ArgumentParser(description="Runs Riemannian linear regression (employing coreset contruction) on the specified dataset")
parser.add_argument('nm', type=str, help="The name of the coreset construction algorithm to use (examples: SVI / GIGAO / GIGAR / RAND)")
parser.add_argument('tr', type=int, help="The trial number - used to initialize random number generation (for replicability)")
parser.add_argument('--M', type=int, default='20', help='Desired maximum coreset size')
parser.add_argument('--proj_dim', type=int, default = '100', help = "The number of samples to take when discretizing log likelihoods")
parser.add_argument('--SVI_opt_itrs', type=int, default = '2000', help = '(If using SVI/HOPS) The number of iterations used when optimizing weights.')
parser.add_argument('--SVI_step_sched', type=str, default = "lambda i : 1.e5/(1+i)", help="Step schedule (tuning rate) for SVI & HOPS, entered as a lambda expression in quotation marks. Default is \"lambda i : 1.e5/(1+i)\"")
parser.add_argument('--pihat_noise', type=float, default=.75, help = "(If calling GIGAR or simulating another realistically tuned Hilbert Coreset) - a measure of how much noise to introduce to the smoothed pi-hat to make the sampler")
parser.add_argument('--n_subsample_opt', type=int, default=500, help="(If using Sparse VI/HOPS) the size of the random subsample to use when optimizing the coreset weights in each reweight step")
parser.add_argument('--n_subsample_select', type=int, default=2000, help="(If using Sparse VI/HOPS) the size of the random subsample to use when determining which point to add to the coreset in each select step")
parser.add_argument('--n_bases_per_scale', type=int, default=50, help="The number of Radial Basis Functions per scale")#TODO: verify help message

arguments = parser.parse_args()
nm = arguments.nm
tr = arguments.tr

#use the trial # as seed
np.random.seed(int(tr))

M = arguments.M
SVI_opt_itrs = arguments.SVI_opt_itrs
n_subsample_opt = arguments.n_subsample_opt
n_subsample_select = arguments.n_subsample_select
proj_dim = arguments.proj_dim
pihat_noise =arguments.pihat_noise
n_bases_per_scale = arguments.n_bases_per_scale
SVI_step_sched = eval(arguments.SVI_step_sched)



#load data and compute true posterior
#each row of x is [lat, lon, price]
print('Loading data')

x = np.load('../data/prices2018.npy')
print('dataset size : ', x.shape)

#log transform the prices
x[:, 2] = np.log10(x[:, 2])

#get empirical mean/std
datastd = x[:,2].std()
datamn = x[:,2].mean()

#bases of increasing size; the last one is effectively a constant
basis_unique_scales = np.array([.2, .4, .8, 1.2, 1.6, 2., 100])
basis_unique_counts = np.hstack((n_bases_per_scale*np.ones(6, dtype=np.int64), 1))

#the dimension of the scaling vector for the above bases
d = basis_unique_counts.sum()
print('Basis dimension: ' + str(d))

#model params
mu0 = datamn*np.ones(d)
Sig0 = 1e-10*(datastd**2+datamn**2)*np.eye(d)
#Sig = datastd**2*np.eye(d)
#SigL = np.linalg.cholesky(Sig)
Sig0inv = np.linalg.inv(Sig0)
#Siginv = np.linalg.inv(Sig)
#SigLInv = np.linalg.inv(SigL)

#generate basis functions by uniformly randomly picking locations in the dataset
print('Trial ' + str(tr)) 
print('Creating bases')
basis_scales = np.array([])
basis_locs = np.zeros((0,2))
for i in range(basis_unique_scales.shape[0]):
  basis_scales = np.hstack((basis_scales, basis_unique_scales[i]*np.ones(basis_unique_counts[i])))
  idcs = np.random.choice(np.arange(x.shape[0]), replace=False, size=basis_unique_counts[i])
  basis_locs = np.vstack((basis_locs, x[idcs, :2]))

print('Converting bases and observations into X/Y matrices')
#convert basis functions + observed data locations into a big X matrix
X = np.zeros((x.shape[0], basis_scales.shape[0]))
for i in range(basis_scales.shape[0]):
  X[:, i] = np.exp( -((x[:, :2] - basis_locs[i, :])**2).sum(axis=1) / (2*basis_scales[i]**2) )
Y = x[:, 2]
Z = np.hstack((X, Y[:,np.newaxis]))
#_, betaV = np.linalg.eigh(X.T.dot(X))
#betaV = betaV[:, -beta_dim:]

#get true posterior
print('Computing true posterior')
mup, LSigp, LSigpInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, Z, np.ones(X.shape[0]))
Sigp = LSigp.dot(LSigp.T)
SigpInv = LSigpInv.dot(LSigpInv.T)

#create function to output log_likelihood given param samples
print('Creating log-likelihood function')
log_likelihood = lambda z, th : model_linreg.log_likelihood(z, th, datastd**2)

print('Creating gradient log-likelihood function')
grad_log_likelihood = lambda z, th : model_linreg.grad_x_log_likelihood(z, th, datastd**2)

#create tangent space for well-tuned Hilbert coreset alg
print('Creating tuned projector for Hilbert coreset construction')
sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSigp.T)
prj_optimal = bc.BlackBoxProjector(sampler_optimal, proj_dim, log_likelihood, grad_log_likelihood)

#create tangent space for poorly-tuned Hilbert coreset alg
print('Creating untuned projector for Hilbert coreset construction')
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)

sampler_realistic = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSighat.T)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, proj_dim, log_likelihood, grad_log_likelihood)

print('Creating black box projector')
def sampler_w(n, wts, pts):
    if pts.shape[0] == 0:
      wts = np.zeros(1)
      pts = np.zeros((1, Z.shape[1]))
    muw, LSigw, LSigwInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, pts, wts)
    return muw + np.random.randn(n, muw.shape[0]).dot(LSigw.T)
prj_w = bc.BlackBoxProjector(sampler_w, proj_dim, log_likelihood, grad_log_likelihood)

#print('Creating exact projectors')
###############################
####Exact projection in SparseVI for gradient computation
##for this model we can do the tangent space projection exactly
#class LinRegProjector(bc.Projector):
#    def __init__(self, bV):
#        self.bV = bV
#
#    def project(self, pts, grad=False):
#        X = pts[:, :-1]
#        Y = pts[:, -1]
#        beta = X.dot(self.V*np.sqrt(np.maximum(self.lmb, 0.)))
#        nu = Y - X.dot(self.muw)
#        #approximation to avoid high memory cost: project the matrix term down to bV.shape[1]**2 dimensions
#        beta_proj = beta.dot(self.bV)
#        return np.hstack((nu[:, np.newaxis]*beta, 1./np.sqrt(2.)*(beta_proj[:, :, np.newaxis]*beta_proj[:, np.newaxis, :]).reshape(beta.shape[0], self.bV.shape[1]**2))) / datastd**2
#
#    def update(self, wts, pts):
#        if pts.shape[0] == 0:
#            self.muw = mu0
#            self.Sigw = Sig0
#        else:
#            self.muw, self.Sigw = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, pts, wts)
#        self.lmb, self.V = np.linalg.eigh(self.Sigw)
#
#prj_exact_optimal = LinRegProjector(betaV)
#prj_exact_optimal.update(np.ones(x.shape[0]), Z)
#rlst_idcs = np.arange(x.shape[0])
#np.random.shuffle(rlst_idcs)
#rlst_idcs = rlst_idcs[:int(0.1*rlst_idcs.shape[0])]
#rlst_w = np.zeros(x.shape[0])
#rlst_w[rlst_idcs] = 2.*x.shape[0]/rlst_idcs.shape[0]*np.random.rand(rlst_idcs.shape[0])
#prj_exact_realistic = LinRegProjector(betaV)
#prj_exact_realistic.update(2.*np.random.rand(x.shape[0]), Z )

##############################


#create coreset construction objects
print('Creating coreset construction objects')
sparsevi = bc.SparseVICoreset(Z, prj_w, opt_itrs = SVI_opt_itrs, n_subsample_opt = n_subsample_opt,  n_subsample_select = n_subsample_select, step_sched = SVI_step_sched)
giga_optimal = bc.HilbertCoreset(Z, prj_optimal)
#giga_optimal_exact = bc.HilbertCoreset(Z, prj_exact_optimal)
giga_realistic = bc.HilbertCoreset(Z, prj_realistic)
#giga_realistic_exact = bc.HilbertCoreset(Z, prj_exact_realistic)
unif = bc.UniformSamplingCoreset(Z)

algs = {'SVI': sparsevi, 
        'GIGAO': giga_optimal, 
        #'GIGAOE': giga_optimal_exact, 
        'GIGAR': giga_realistic, 
        #'GIGARE': giga_realistic_exact, 
        'RAND': unif}
alg = algs[nm]

print('Building coreset')
#build coresets
w = [np.array([0.])]
p = [np.zeros((1, Z.shape[1]))]
cputs = [0] #TODO: verify that it's fair to think of the time before build calls as effectively 0
tot_time = 0
for m in range(1, M+1):
  print('trial: ' + str(tr) +' alg: ' + nm + ' ' + str(m) +'/'+str(M))
  t0 = time.process_time()
  alg.build(1)
  tot_time += time.process_time()-t0
  #store weights
  wts, pts, idcs = alg.get()
  w.append(wts)
  p.append(pts)
  cputs.append(tot_time)

  #printouts for debugging purposes
  #print('reverse KL: ' + str(model_linreg.weighted_post_KL(mu0, Sig0inv, datastd**2, Z, w[m, :], reverse=True)))

muw = np.zeros((M+1, mu0.shape[0]))
Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
rklw = np.zeros(M+1)
fklw = np.zeros(M+1)
for m in range(M+1):
  print('KL divergence computation for trial: ' + str(tr) +' alg: ' + nm + ' ' + str(m) +'/'+str(M))
  muw[m, :], LSigw, LSigwInv = model_linreg.weighted_post(mu0, Sig0inv, datastd**2, p[m], w[m])
  Sigw[m, :, :] = LSigw.dot(LSigw.T)
  rklw[m] = model_linreg.KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
  fklw[m] = model_linreg.KL(mup, Sigp, muw[m,:], LSigwInv.dot(LSigwInv.T))

if not os.path.exists('results/'):
  os.mkdir('results')
print('Saving result for trial: ' + str(tr) +' alg: ' + nm)
SVI_step_sched_hash_sha1 = hashlib.sha1(arguments.SVI_step_sched.encode('utf-8')).hexdigest()

np.savez_compressed(os.path.join('results/', nm+'_'+'tr='+str(tr)+'_n_subsample_opt='+str(n_subsample_opt)+
'_n_subsample_select='+str(n_subsample_select)+'_'+'SVI_step_sched_hash_sha1='+SVI_step_sched_hash_sha1+
'_pihat_noise='+str(pihat_noise)+'.npz'),cputs=cputs, mu0= mu0, Sig0=Sig0, mup=mup, Sigp=Sigp, w=w, p=p, muw=muw, Sigw=Sigw,
rklw=rklw, fklw=fklw, basis_scales = basis_scales, basis_locs=basis_locs, datastd=datastd, nm=nm, tr=tr,
n_bases_per_scale=n_bases_per_scale, proj_dim=proj_dim, SVI_opt_itrs=SVI_opt_itrs, n_subsample_opt=n_subsample_opt, 
n_subsample_select=n_subsample_select, SVI_step_sched=arguments.SVI_step_sched, pihat_noise=pihat_noise)