from __future__ import print_function
import numpy as np
import bayesiancoresets as bc
from scipy.optimize import minimize
import time
import sys
from scipy.optimize import nnls
#np.seterr(over='raise', invalid='raise', divide='raise')

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
  Zw = Z[wts>0, :]
  ww = wts[wts>0]
  while True:
    try:
      res = minimize(lambda mu : -log_joint(Zw, mu, ww), mu0, jac=lambda mu : -grad_log_joint(Zw, mu, ww))
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
  Sig = -np.linalg.inv(hess_log_joint_w(Zw, mu, ww))
  return mu, Sig

#performs selection of the next datapoint to add to the riemann coreset
def riemann_select(HlogZ1w, lls):
  stds = lls.std(axis=1)
  corrs = (lls*HlogZ1w)
  return 
  
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
  
##computes the gradient w.r.t. alpha, beta in the single-weight-update greedy method
#def grad_line(ab, Z, w, one_n, lls):
#  alpha = ab[0]
#  beta = ab[1]
#  wab = beta*(w+alpha*one_n)
#  #compute gradients
#  one_f = lls.sum(axis=0)
#  wab_f = wab.dot(lls)
#  dKLdb = -1./beta*(wab_f*(one_f-wab_f)).mean()
#  dKLda = -beta*(lls[n,:]*(one_f-wab_f)).mean()
#  return np.array([dKLda, dKLdb])

#runs the single-weight-update optimization for greedy
def riemann_optimize_line(w, n, HlogZ1w, HlogZa, H3logZa):
  one_n = np.zeros(w.shape[0])
  one_n[n] = 1.
  grad = lambda x, itr : grad_line(x, Z, w, one_n, lls)
  res = minimize(lambda mu : -log_joint(Zw, mu, ww), mu0, jac=lambda mu : -grad_log_joint(Zw, mu, ww))


  ab = adam(grad, np.array([0., 1.]), adam_num_iters, adam_learning_rate)
  return ab[1]*(w+ab[0]*one_n)

##computes the gradient w.r.t. w (active indices only) for fully corrective greedy
#def grad_full(w, Z, lls, active_idcs):
#  #compute residual error
#  residuals = lls.sum(axis=0) - w.dot(lls) 
#  #compute gradient
#  dKLdw = np.zeros(w.shape[0])
#  dKLdw[active_idcs] = (lls[active_idcs, :]*residuals).mean(axis=1)
#  return dKLdw

#runs the full weight reoptimization
def riemann_optimize(w, n,  HlogZ1w, HlogZa, H3logZa, full=True):
  lmb, V = np.linalg.eigh(HlogZa - H3logZa+1e-16*np.eye(HlogZa.shape[0]))
  eta = 1.
  while np.any(lmb <= 0.):
    eta /= 2.
    lmb, V = np.linalg.eigh(HlogZa - eta*H3logZa+1e-16*np.eye(HlogZa.shape[0]))
  one_n = np.zeros(w.shape[0])
  one_n[n] = 1.
  C = (V*np.sqrt(lmb)).T
  Cinv = (V/np.sqrt(lmb))
  B = C.dot(w) + Cinv.T.dot(HlogZ1w)
  if full:
    A = C
    w, resid = nnls(A,B) 
  else:
    A = np.atleast_2d(np.hstack((C.dot(one_n[:,np.newaxis]), C.dot(w[:,np.newaxis]))))
    x, resid = nnls(A,B) 
    w = x[1]*w+x[0]*one_n
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
  samples = np.hstack((samples[:, 1:], samples[:, 0][:,np.newaxis]))
  # THESE WORK FOR RIEMANN_CORR
  tuning = {'synth': (1000, lambda itr : 10./(1.+itr)), 'ds1': (2000, lambda itr : 10./(1.+itr)), 'phishing': (2000, lambda itr : 10./(1.+itr)**0.8)}
else:
  from model_poiss import *
  print('Loading dataset '+dnm)
  Z, Zt, D = load_data('poiss/'+dnm+'.npz')
  print('Loading posterior samples for '+dnm)
  samples = np.load('poiss_'+dnm+'_samples.npy')
  #need to put intercept at the end
  samples = np.hstack((samples[:, 1:], samples[:, 0][:,np.newaxis]))
  # THESE WORK FOR RIEMANN_CORR
  #tuning = {'synth':(10., 700, 1.8,8000), 'biketrips':(1., 100, 2, 32000), 'airportdelays':(1., 100., 2., 32000)}
  tuning = {'synth': (1000, lambda itr : 10./(1.+itr)), 'biketrips': (2000, lambda itr : 5./(1.+itr)**0.8), 'airportdelays': (2000, lambda itr : 4./(1.+itr)**0.75)}

#fit a gaussian to the posterior samples 
#used for pihat computation for Hilbert coresets with noise to simulate uncertainty in a good pihat
mup = samples.mean(axis=0)
Sigp = np.cov(samples, rowvar=False)

#create the prior -- also used for the above purpose
mu0 = np.zeros(mup.shape[0])
Sig0 = np.eye(mup.shape[0])

#learning_rate=tuning[dnm][0]
#n_samples=tuning[dnm][1]
#eta=tuning[dnm][2]
#n_samples_max = tuning[dnm][3]
n_samples = tuning[dnm][0]
learning_rate = tuning[dnm][1]

#print('learning rate ' + str(learning_rate))
#print('n_samples ' + str(n_samples))
###############################
## TUNING PARAMETERS ##
Ms = [1, 2, 5, 10, 20, 50, 100, 500]# , 1000] #coreset sizes at which we record output
projection_dim = 100 #random projection dimension for Hilbert csts
pihat_noise = 0.15 #noise level (relative) for corrupting pihat
#adam_num_iters = 200 #number of ADAM optimization iterations in riemann csts
#adam_learning_rate = lambda itr : 1./np.sqrt(itr+1.) #ADAM learning rate in riemann csts
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
  lls -= lls.mean(axis=1)[:,np.newaxis]
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
elif alg == 'hilbert_good' or alg == 'hilbert_corr_good':
  #here we assume we have the true posterior samples
  proj_samps = np.random.multivariate_normal(mup, Sigp, projection_dim)
  #compute random projection
  lls = np.zeros((Z.shape[0], projection_dim))
  for i in range(proj_samps.shape[0]):
    lls[:, i] = log_likelihood(Z, proj_samps[i, :])
  lls -= lls.mean(axis=1)[:,np.newaxis]
  #Build coreset via GIGA
  giga = bc.GIGACoreset(lls)
  for m in range(len(Ms)):
    print(str(m+1)+'/'+str(len(Ms)))
    giga.build(Ms[m])
    #if we want to fully reoptimize in each step, call giga.optimize()
    if alg == 'hilbert_corr_good':
      giga.optimize() 
    #record time and weights
    cputs[m] = time.process_time()-t0
    wts[m, :] = giga.weights()
elif alg == 'riemann' or alg == 'riemann_corr':
  #normal dist for approx piw sampling; will use laplace throughout
  w = np.zeros(Z.shape[0])
  muw = np.random.randn(mu0.shape[0])
  
  for m in range(len(Ms)):
    print('keypoint ' + str(m+1) + ' / ' + str(len(Ms))+': M = ' + str(Ms[m]))
    #n_samples*=eta
    #n_samples = int(n_samples)
    #n_samples = min(n_samples, n_samples_max)
    #print('setting n_samples = ' + str(n_samples))
    
    std_samps = np.random.randn(n_samples, mu0.shape[0])
    lls = np.zeros((Z.shape[0], n_samples))
    #build up to Ms[m] one point at a time
    for j in range(Ms[m]-Ms[m-1] if m>0 else Ms[m]):
      if j % 10 == 0:
        print('j = ' + str(j+1) + '/' + str(Ms[m]-Ms[m-1] if m>0 else Ms[m]))
      print('n_samples = ' + str(n_samples))

      #get laplace w-posterior approx for sampling
      if (w>0).sum()>0:
        muw, Sigw = get_laplace(w, Z, muw)
      else:
        muw, Sigw = mu0, Sig0

      print(float(Z.shape[0]))
      print(w[w>0])
      mn_err = np.sqrt(((muw-mup)**2).sum())/np.sqrt(((mup**2).sum()))
      cv_err = np.sqrt(((Sigw-Sigp)**2).sum())/np.sqrt(((Sigp)**2).sum())
      print('mean error : ' + str(mn_err)+ '\n covar error: ' + str(cv_err))
      #print('mean : ' + str(muw)+ '\n covar: ' + str(np.diagonal(Sigw)))
      #print('mean true : ' + str(mup)+ '\n covar true: ' + str(np.diagonal(Sigp)))

      samps = muw + np.linalg.cholesky(Sigw).dot(std_samps.T).T
      #samps = np.random.multivariate_normal(muw, Sigw, n_samples)

      #compute a finite dimension projection of the tangent hilbert space
      #compute lls at this w
      for i in range(n_samples):
        lls[:, i] = log_likelihood(Z, samps[i,:])
      #subtract off the mean
      lls -= lls.mean(axis=1)[:, np.newaxis]

      #select next datapoint
      #compute HlogZ(1-w), stds, and correlations
      lls_1w = np.dot(np.ones(w.shape[0]) - w, lls)

      HlogZ1w = (lls*lls_1w).mean(axis=1)
      stds = lls.std(axis=1)
      #compute correlations (w/o normalizing residual, since it doesn't affect selection)
      corrs = HlogZ1w/stds
      #for data in the active set, we look at abs(corr); for nonactive, only positive correlations are good
      corrs[w>0] = np.fabs(corrs[w>0])
      n = corrs.argmax()

      #compute hessian and third derive*(1-w) at active idcs
      active_idcs = w>0
      active_idcs[n] = True
      n_active = (np.cumsum(active_idcs)-1)[n]
 
      lls_active = np.atleast_2d(lls[active_idcs, :])
      HlogZa = lls_active.dot(lls_active.T)/n_samples
      H3logZa = (lls_1w*lls_active).dot(lls_active.T)/n_samples
      #H3logZa /= (1.+learning_rate/(j+1+(Ms[m-1] if m > 0 else 0)))
      print('MAGNITUDES')
      print(np.fabs(HlogZa).mean())
      print(np.fabs(H3logZa).mean())
      print(np.fabs(HlogZ1w).mean())
      wprev = w.copy()
      #optimize the weights
      w_new  = riemann_optimize(w[active_idcs], n_active, HlogZ1w[active_idcs], HlogZa, H3logZa, full= (alg == 'riemann_corr'))
      gamma = learning_rate(j + (Ms[m]-Ms[m-1] if m>0 else 0))
      w[active_idcs] = (1. - gamma)*w[active_idcs] + gamma*w_new
      ##threshold to make sure things aren't getting too crazy; no datapoint should be weighted more than the entire dataset
      #w[active_idcs] = np.random.multinomial(Z.shape[0], np.ones(active_idcs.sum())/(active_idcs.sum()))
      #w[w > Z.shape[0]] = Z.shape[0]
    #record the weights and cput time
    wts[m, :] = w.copy()
    #record time
    cputs[m] = time.process_time()-t0
elif alg == 'uniform':
  print(str(1)+'/'+str(len(Ms)))
  cts = np.zeros(wts.shape[1])
  cts = np.random.multinomial(Ms[0], np.ones(Z.shape[0])/float(Z.shape[0]))
  wts[0, :] = cts*Z.shape[0]/cts.sum()
  cputs[0] = time.process_time() - t0
  for m in range(1, len(Ms)):
    print(str(m+1)+'/'+str(len(Ms)))
    cts += np.random.multinomial(Ms[m]-Ms[m-1], np.ones(Z.shape[0])/float(Z.shape[0]))
    wts[m, :] = cts*Z.shape[0]/cts.sum()
    #record time
    cputs[m] = time.process_time() - t0
elif alg == 'prior':
  #do nothing to weights and cputs, always prior + 0
  pass

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


