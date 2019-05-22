from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
import time
import sys

np.seterr(over='raise', invalid='raise', divide='raise')

#adam optimizer with lambda fcn learning rate -- pulled from autograd library
def adam(grad, x, num_iters, learning_rate, 
        b1=0.9, b2=0.999, eps=10**-8,callback=None):
    """Adam as described in http://arxiv.org/pdf/1412.6980.pdf.
    It's basically RMSprop with momentum and some correction terms."""
    m = np.zeros(len(x))
    v = np.zeros(len(x))
    for i in range(num_iters):
        g = grad(x, i)
        if callback: callback(x, i, g)
        m = (1 - b1) * g      + b1 * m  # First  moment estimate.
        v = (1 - b2) * (g**2) + b2 * v  # Second moment estimate.
        mhat = m / (1 - b1**(i + 1))    # Bias correction.
        vhat = v / (1 - b2**(i + 1))
        x = x - learning_rate(i)*mhat/(np.sqrt(vhat) + eps)
    return x

#computes KL( N(mu0, Sig0) || N(mu1, Sig1) )
def gaussian_KL(mu0, Sig0, mu1, Sig1):
  t1 = np.linalg.solve(Sig1, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.linalg.solve(Sig1, mu1-mu0))
  t3 = np.linalg.slogdet(Sig1)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])


#computes the Laplace approximation N(mu, Sig) to the posterior with weights wts
def get_laplace(wts, Z, mu0):
  trials = 10
  while True:
    try:
      res = minimize(lambda mu : -log_joint(Z, mu, wts), mu0, jac=lambda mu : -grad_log_joint(Z, mu, wts))
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
  Sig = -np.linalg.inv(hess_log_joint(Z, mu))
  return mu, Sig

#performs selection of the next datapoint to add to the riemann coreset
def riemann_select(Z, w, muw, Sigw, n_samples):
  #take samples for empirical correlation estimation 
  samps = np.random.multivariate_normal(muw, Sigw, n_samples)
  #compute log likelihoods
  lls = np.zeros((Z.shape[0], n_samples))
  for i in range(n_samples):
    lls[:, i] = log_likelihood(Z, samps[i,:])
  #subtract off the mean
  lls -= lls.mean(axis=1)[:, np.newaxis]
  #compute residual error
  residuals = lls.sum(axis=0) - w.dot(lls) 
  #get std dev of lls
  stds = lls.std(axis=1)
  #compute correlations (w/o normalizing residual, since it doesn't affect selection)
  corrs = (lls*residuals).mean(axis=1)/stds
  #for data in the active set, we look at abs(corr); for nonactive, only positive correlations are good
  corrs[w>0] = np.fabs(corrs[w>0])
  return corrs.argmax()
  
#computes the gradient w.r.t. alpha, beta in the single-weight-update greedy method
def grad_line(ab, Z, w, one_n, n_samps, muw):
  alpha = ab[0]
  beta = ab[1]
  #get samples from pi_b*(w+a1n)
  mu, Sig =  get_laplace(beta*(w+alpha*one_n), Z, muw)
  samps = np.random.multivariate_normal(mu, Sig, n_samps)
  #compute log likelihoods
  lls = np.zeros((Z.shape[0], n_samples))
  for i in range(n_samples):
    lls[:, i] = log_likelihood(Z, samps[i,:])
  #subtract off the mean
  lls -= lls.mean(axis=1)[:, np.newaxis]
  wab = beta*(w+alpha*one_n)
  #compute gradients
  one_f = lls.sum(axis=0)
  wab_f = wab.dot(lls)
  dKLdb = -1./beta*(wab_f*(one_f-wab_f)).mean()
  dKLda = -beta*(lls[n,:]*(one_f-wab_f)).mean()
  return np.array([dKLda, dKLdb])

#runs the single-weight-update optimization for greedy
def riemann_optimize_line(Z, w, n, muw, n_samples, adam_num_iters, adam_learning_rate):
  one_n = np.zeros(Z.shape[0])
  one_n[n] = 1.
  grad = lambda x, itr : grad_line(x, Z, w, one_n, n_samples, muw)
  ab = adam(grad, np.array([0., 1.]), adam_num_iters, adam_learning_rate)
  return ab[1]*(w+ab[0]*one_n)

#computes the gradient w.r.t. w (active indices only) for fully corrective greedy
def grad_full(w, Z, n_samps, active_idcs, muw):
  #get samples from pi_w
  mu, Sig =  get_laplace(w, Z, muw)
  samps = np.random.multivariate_normal(mu, Sig, n_samps)
  #compute log likelihoods
  lls = np.zeros((Z.shape[0], n_samples))
  for i in range(n_samples):
    lls[:, i] = log_likelihood(Z, samps[i,:])
  #subtract off the mean
  lls -= lls.mean(axis=1)[:, np.newaxis]
  #compute residual error
  residuals = lls.sum(axis=0) - w.dot(lls) 
  #compute gradient
  dKLdw = np.zeros(w.shape[0])
  dKLdw[active_idcs] = (lls[active_idcs, :]*residuals).mean(axis=1)
  return dKLdw

#runs the full weight reoptimization
def riemann_optimize_full(Z, w, n, muw, n_samples, adam_num_iters, adam_learning_rate):
  active_idcs = w>0
  active_idcs[n] = True
  grad = lambda x, itr : grad_full(x, Z, n_samples, active_idcs, muw)
  w = adam(grad, w, adam_num_iters, adam_learning_rate)
  return w


fldr = sys.argv[1] #should be either lr or poiss
dnm = sys.argv[2] #if above is lr, should be synth / phishing / ds1; if above is poiss, should be synth, biketrips, or airportdelays
alg = sys.argv[3] #should be hilbert / hilbert_corr / riemann / riemann_corr / uniform 
ID = sys.argv[4] #just a number to denote trial #, any nonnegative integer

#e.g.  python3 main.py lr phishing riemann_corr 2 
#  will run fully corrective riemann coresets on logistic regression with the phishing dataset, trial # 2


#load the logistic or poisson regression model depending on selected folder
if fldr == 'lr':
  from model_lr import *
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data('lr/'+dnm+'.npz')
  print('Loading posterior samples for '+dnm)
  samples = np.load('lr_'+dnm+'_samples.npy')
else:
  from model_poiss import *
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data('poiss/'+dnm+'.npz')
  print('Loading posterior samples for '+dnm)
  samples = np.load('poiss_'+dnm+'_samples.npy')

#fit a gaussian to the posterior samples 
#used for pihat computation for Hilbert coresets with noise to simulate uncertainty in a good pihat
mup = samples.mean(axis=0)
Sigp = np.cov(samples, rowvar=False)
#create the prior -- also used for the above purpose
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

###############################
## TUNING PARAMETERS ##
Ms = [1, 2, 5, 10, 20, 50, 100] #coreset sizes at which we record output
projection_dim = 500 #random projection dimension for Hilbert csts
pihat_noise = 0.15 #noise level (relative) for corrupting pihat
n_samples = 20 #number of samples for KL gradients in ADAM optimization for riemann csts
adam_num_iters = 10000 #number of ADAM optimization iterations in riemann csts
adam_learning_rate = lambda itr : 1./np.sqrt(itr+1.) #ADAM learning rate in riemann csts
###############################

#initialize memory for coreset weights, laplace approx, kls
wts = np.zeros((len(Ms), Z.shape[0]))
cputs = np.zeros(len(Ms))
print('Building coresets via ' + alg)
t0 = time.process_time()
if alg == 'hilbert' or alg == 'hilbert_corr':
  #get pihat via interpolation between prior/posterior + noise
  #uniformly smooth between prior and posterior
  U = np.random.rand()
  muhat = U*mup + (1.-U)*mu0
  Sighat = U*Sigp + (1.-U)*Sig0
  #now corrupt the smoothed pihat
  muhat += pihat_noise*np.sqrt((muhat**2).sum())*np.random.randn(muhat.shape[0])
  Sighat *= np.exp(2*pihat_noise*np.random.randn())
  #take pihat samples for random projection
  proj_samps = np.random.multivariate_normal(muhat, Sighat, projection_dim)
  #compute random projection
  lls = np.zeros((Z.shape[0], projection_dim))
  for i in range(proj_samps.shape[0]):
    lls[:, i] = log_likelihood(Z, proj_samps[i, :])
  try:
    lls -= lls.mean(axis=1)[:,np.newaxis]
  except:
    print(np.isinf(lls))
  #Build coreset via GIGA
  giga = bc.GIGACoreset(lls)
  for m in range(len(Ms)):
    print(str(m+1)+'/'+str(len(Ms)))
    giga.build(Ms[m])
    #if we want to fully reoptimize in each step, call giga.optimize()
    if alg == 'hilbert_corr':
      giga.optimize() 
    #record time and weights
    cputs[m] = time.process_time()-t0
    wts[m, :] = giga.weights()
elif alg == 'riemann' or alg == 'riemann_corr':
  #normal dist for approx piw sampling; will use laplace throughout
  w = np.zeros(Z.shape[0])
  muw = np.random.randn(Z.shape[1])
  for m in range(len(Ms)):
    #build up to Ms[m] one point at a time
    for j in range(Ms[m]-Ms[m-1] if m>0 else Ms[m]):
      #get laplace w-posterior approx for sampling
      muw, Sigw = get_laplace(w, Z, muw)
      #select next datapoint
      n = riemann_select(Z, w, muw, Sigw, n_samples)
      #optimize the weights
      if alg == 'riemann_corr':
        w = riemann_optimize_full(Z, w, n, muw, n_samples, adam_num_iters, adam_learning_rate)
      else:
        w = riemann_optimize_line(Z, w, n, muw, n_samples, adam_num_iters, adam_learning_rate)
    #record the weights and cput time
    wts[m, :] = w.copy()
    #record time
    cputs[m] = time.process_time()-t0
elif alg == 'uniform':
  print(str(1)+'/'+str(len(Ms)))
  wts[0, :] = np.random.multinomial(Ms[0], np.ones(Z.shape[0])/float(Z.shape[0]))
  cputs[0] = time.process_time() - t0
  for m in range(1, len(Ms)):
    print(str(m+1)+'/'+str(len(Ms)))
    wts[m, :] = wts[m-1, :] + np.random.multinomial(Ms[m]-Ms[m-1], np.ones(Z.shape[0])/float(Z.shape[0]))
    #record time
    cputs[m] = time.process_time() - t0

#get laplace approximations for each weight setting, and KL divergence to full posterior laplace approx mup Sigp
#used for a quick/dirty performance comparison without expensive posterior sample comparisons (e.g. energy distance)
mus_laplace = np.zeros((len(Ms), D))
Sigs_laplace = np.zeros((len(Ms), D, D))
kls_laplace = np.zeros(len(Ms))
print('Computing coreset Laplace approximation + approximate KL(posterior || coreset laplace)')
for m in range(len(Ms)):
  mul, Sigl = get_laplace(wts[m,:], Z, Z.mean(axis=0)[:D])
  mus_laplace[m,:] = mul
  Sigs_laplace[m,:,:] = Sigl
  kls_laplace[m] = gaussian_KL(mup, Sigp, mus_laplace[m,:], Sigs_laplace[m,:,:])

#save results
np.savez(fldr+'_'+dnm+'_'+alg+'_results_'+str(ID)+'.npz', cputs=cputs, wts=wts, Ms=Ms, mus=mus_laplace, Sigs=Sigs_laplace, kls=kls_laplace)


