import numpy as np
import scipy.linalg as sl
import pickle as pk
import os, sys
from scipy.stats import multivariate_normal
#make it so we can import models/etc from parent folder
hpc=True
if hpc: sys.path.insert(1, os.path.join(sys.path[0], '/home/dm754/bayesian-coresets-private'))
import bayesiancoresets as bc
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import gaussian

M = 200
N = 1000
SVI_opt_itrs = 500
BPSVI_opt_itrs = 500
n_subsample_opt = None # 100
proj_dim = 100
pihat_noise =0.75
BPSVI_step_sched = lambda m: lambda i : (-0.005*m+1.005)/(1+i) # linear interpolaton giving i0=1 at m=1 and i0=0.005 at m=200 
SVI_step_sched = lambda i : 1./(1+i)

def linearize():
  args_dict = dict()
  c = -1
  for tr in range(0,10): # trial number
    for nm in ["BPSVI"]: # ["RAND", "BPSVI", "SVI", "GIGAO", "GIGAR"]: # coreset method
      for d in [200]: # [200, 500]: # data dimension
          #for i0 in [5., 1.5, 0.5, 0.1]: # initial learning rate
          c += 1
          args_dict[c] = (tr, nm, d) #(tr, nm, d, i0)
  return args_dict

mapping = linearize()

#print('mapping : ', len(mapping))
#print( 'mapping at 0 : ', mapping[0] )
#exit()
tr, nm, d = mapping[int(sys.argv[1])]
#tr, nm, d = mapping[0]


mu0 = np.zeros(d)
Sig0 = np.eye(d)
Sig = np.eye(d)
SigL = np.linalg.cholesky(Sig)
th = np.ones(d)
Sig0inv = np.linalg.inv(Sig0)
Siginv = np.linalg.inv(Sig)
SigLInv = np.linalg.inv(SigL)
logdetSig = np.linalg.slogdet(Sig)[1]

#generate data and compute true posterior
#use the trial # as the seed
np.random.seed(int(tr))

print('Computing true posterior')
x = np.random.multivariate_normal(th, Sig, N)
mup, LSigp, LSigpInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
Sigp = LSigp.dot(LSigp.T)
SigpInv = LSigpInv.dot(LSigpInv.T)

#for the algorithm, use the trial # and name as seed
np.random.seed(int(''.join([ str(ord(ch)) for ch in nm+str(tr)])) % 2**32)

#create the log_likelihood function
print('Creating log-likelihood function')
log_likelihood = lambda x, th : gaussian.gaussian_loglikelihood(x, th, Siginv, logdetSig)

print('Creating gradient log-likelihood function')
grad_log_likelihood = lambda x, th : gaussian.gaussian_gradx_loglikelihood(x, th, Siginv)

print('Creating tuned projector for Hilbert coreset construction')
#create the sampler for the "optimally-tuned" Hilbert coreset
sampler_optimal = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSigp.T)
prj_optimal = bc.BlackBoxProjector(sampler_optimal, proj_dim, log_likelihood, grad_log_likelihood)

print('Creating untuned projector for Hilbert coreset construction')
#create the sampler for the "realistically-tuned" Hilbert coreset
U = np.random.rand()
muhat = U*mup + (1.-U)*mu0
Sighat = U*Sigp + (1.-U)*Sig0
#now corrupt the smoothed pihat
muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
Sighat *= np.exp(-2*pihat_noise*np.fabs(np.random.randn()))
LSighat = np.linalg.cholesky(Sighat)

sampler_realistic = lambda n, w, pts : mup + np.random.randn(n, mup.shape[0]).dot(LSighat.T)
prj_realistic = bc.BlackBoxProjector(sampler_realistic, proj_dim, log_likelihood, grad_log_likelihood)

print('Creating exact projectors')
#exact (gradient) log likelihood projection
class GaussianProjector(bc.Projector):
  def project(self, pts, grad=False):
    nu = (pts - self.muw).dot(SigLInv.T)
    PsiL = SigLInv.dot(self.LSigw)
    Psi = PsiL.dot(PsiL.T)
    nu = np.hstack((nu.dot(PsiL), np.sqrt(0.5*np.trace(np.dot(Psi.T, Psi)))*np.ones(nu.shape[0])[:,np.newaxis]))
    nu *= np.sqrt(nu.shape[1])
    if not grad:
      return nu
    else:
      gnu = np.hstack((SigLInv.T.dot(PsiL), np.zeros(pts.shape[1])[:,np.newaxis])).T
      gnu = np.tile(gnu, (pts.shape[0], 1, 1))
      gnu *= np.sqrt(gnu.shape[1])
      return nu, gnu
  def update(self, wts = None, pts = None):
    if wts is None or pts is None or pts.shape[0] == 0:
      wts = np.zeros(1)
      pts = np.zeros((1, mu0.shape[0]))
    self.muw, self.LSigw, self.LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, pts, wts)

prj_exact_optimal = GaussianProjector()
prj_exact_optimal.update(np.ones(x.shape[0]), x)
rlst_idcs = np.arange(x.shape[0])
np.random.shuffle(rlst_idcs)
rlst_idcs = rlst_idcs[:int(0.1*rlst_idcs.shape[0])]
rlst_w = np.zeros(x.shape[0])
rlst_w[rlst_idcs] = 2.*x.shape[0]/rlst_idcs.shape[0]*np.random.rand(rlst_idcs.shape[0])
prj_exact_realistic = GaussianProjector()
prj_exact_realistic.update(2.*np.random.rand(x.shape[0]), x)

##############################
print('Creating coreset construction objects')
#create coreset construction objects
bpsvi = bc.BatchPSVICoreset(x, GaussianProjector(), opt_itrs = BPSVI_opt_itrs, n_subsample_opt = n_subsample_opt, step_sched = BPSVI_step_sched)
sparsevi = bc.SparseVICoreset(x, GaussianProjector(), opt_itrs = SVI_opt_itrs, step_sched = SVI_step_sched)
giga_optimal = bc.HilbertCoreset(x, prj_optimal)
giga_optimal_exact = bc.HilbertCoreset(x,prj_exact_optimal)
giga_realistic = bc.HilbertCoreset(x,prj_realistic)
giga_realistic_exact = bc.HilbertCoreset(x,prj_exact_realistic)
unif = bc.UniformSamplingCoreset(x)

algs = {'BPSVI' : bpsvi,
        'SVI': sparsevi, 
        'GIGAO': giga_optimal, 
        'GIGAR': giga_realistic, 
        'RAND': unif}
alg = algs[nm]

print('Building coreset')
w = [np.array([0.])]
p = [np.zeros((1, x.shape[1]))]

def build_for_m(m): # auxiliary function for parallelizing BPSVI experiment
  print('trial: ' + str(tr) +' alg: BPSVI ' + str(m) +'/'+str(M))
  alg.build(1, m)
  return alg.get()

if nm=="BPSVI": #parallelize over batch pseudocoreset sizes
  from multiprocessing import Pool
  pool = Pool(processes=64)
  res = pool.map(build_for_m, range(1, M+1))
  for wts, pts, _ in res:
    w.append(wts)
    p.append(pts)
else:
  for m in range(1, M+1):
    print('trial: ' + str(tr) +' alg: ' + nm + ' ' + str(m) +'/'+str(M))
    alg.build(1, m)
    #store weights/pts
    wts, pts, _ = alg.get()
    w.append(wts)
    p.append(pts)

# computing kld and saving results
muw = np.zeros((M+1, mu0.shape[0]))
Sigw = np.zeros((M+1,mu0.shape[0], mu0.shape[0]))
rklw = np.zeros(M+1)
fklw = np.zeros(M+1)
for m in range(M+1):
  muw[m, :], LSigw, LSigwInv = gaussian.weighted_post(mu0, Sig0inv, Siginv, p[m], w[m])
  Sigw[m, :, :] = LSigw.dot(LSigw.T)
  rklw[m] = gaussian.gaussian_KL(muw[m,:], Sigw[m,:,:], mup, SigpInv)
  fklw[m] = gaussian.gaussian_KL(mup, Sigp, muw[m,:], LSigwInv.dot(LSigwInv.T))

if not os.path.exists('results/'):
  os.mkdir('results')
#f = open('results/results_'+nm+'_'+str(d)+'_'+'lr'+'_'+str(i0)+'_'+str(tr)+'.pk', 'wb')
f = open('results/results_adaptr_'+nm+'_'+str(d)+'_'+str(tr)+'.pk', 'wb')
res = (x, mu0, Sig0, Sig, mup, Sigp, w, p, muw, Sigw, rklw, fklw)
pk.dump(res, f)
f.close()
