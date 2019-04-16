import bayesiancoresets as bc
import autograd.numpy as np
from autograd import grad
import warnings
import sys
import cProfile

def gaussian_KL(mu0, Sig0, mu1, Sig1inv):
  t1 = np.dot(Sig1inv, Sig0).trace()
  t2 = np.dot((mu1-mu0),np.dot(Sig1inv, mu1-mu0))
  t3 = -np.linalg.slogdet(Sig1inv)[1] - np.linalg.slogdet(Sig0)[1]
  return 0.5*(t1+t2+t3-mu0.shape[0])

def weighted_post(th0, Sig0inv, Siginv, x, w): 
  Sigp = np.linalg.inv(Sig0inv + w.sum()*Siginv)
  mup = np.dot(Sigp,  np.dot(Sig0inv,th0) + np.dot(Siginv, (w[:, np.newaxis]*x).sum(axis=0)))
  return mup, Sigp

def weighted_post_KL(th0, Sig0inv, Siginv, x, w, reverse=True):
  muw, Sigw = weighted_post(th0, Sig0inv, Siginv, x, w)
  mup, Sigp = weighted_post(th0, Sig0inv, Siginv, x, np.ones(x.shape[0]))
  if reverse:
    return gaussian_KL(muw, Sigw, mup, np.linalg.inv(Sigp))
  else:
    return gaussian_KL(mup, Sigp, muw, np.linalg.inv(Sigw))

#NB: without constant terms
#E[Log N(x; mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m1_exact(muw, Sigw, Siginv, x):
  return -0.5*np.trace(np.dot(Siginv, Sigw)) -0.5*(np.dot((x - muw), Siginv)*(x-muw)).sum(axis=1)

#Covar[Log N(x; mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m2_exact(muw, Sigw, Siginv, x):
  L = np.linalg.cholesky(Siginv)
  Rho = np.dot(np.dot(L.T, Sigw), L)

  crho = 2*(Rho**2).sum() + (np.diag(Rho)*np.diag(Rho)[:, np.newaxis]).sum()

  mu = np.dot(L.T, (x - muw).T).T
  musq = (mu**2).sum(axis=1)

  return 0.25*(crho + musq*musq[:, np.newaxis] + np.diag(Rho).sum()*(musq + musq[:,np.newaxis]) + 4*np.dot(np.dot(mu, Rho), mu.T))

#Var[Log N(x;, mu, Sig)] under mu ~ N(muw, Sigw)
def ll_m2_exact_diag(muw, Sigw, Siginv, x):
  L = np.linalg.cholesky(Siginv)
  Rho = np.dot(np.dot(L.T, Sigw), L)

  crho = 2*(Rho**2).sum() + (np.diag(Rho)*np.diag(Rho)[:, np.newaxis]).sum()

  mu = np.dot(L.T, (x - muw).T).T
  musq = (mu**2).sum(axis=1)

  return 0.25*(crho + musq**2 + 2*np.diag(Rho).sum()*musq + 4*(np.dot(mu, Rho)*mu).sum(axis=1))
  
class ExactGaussianL1KLCoreset(bc.L1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, reverse=True, scaled=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N = x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=reverse, scaled=scaled)

  def _compute_scales(self):
    return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=False)

  def _reverse_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=True)

  def _forward_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=False))
    return g(w)

  def _reverse_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=True))
    if normalize:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)

class EGL1Reverse(ExactGaussianL1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig): 
    super().__init__(x, mu0, Sig0, Sig, True) 

class EGL1Forward(ExactGaussianL1KLCoreset):
  def __init__(self, x, mu0, Sig0, Sig):
    super().__init__(x, mu0, Sig0, Sig, False) 

class ExactGaussianGreedyKLCoreset(bc.GreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig, reverse=True):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(N=x.shape[0], potentials=None, sampler=None, n_samples=None, reverse=reverse)

  #def _compute_scales(self):
  #  return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=False)

  def _reverse_kl(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=True)

  def _forward_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=False))
    return g(w)

  def _reverse_kl_grad(self, w, normalize):
    g = grad(lambda w : weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=True))
    if normalize:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)

class EGGreedyReverse(ExactGaussianGreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig): 
    super().__init__(x, mu0, Sig0, Sig, True) 

class EGGreedyForward(ExactGaussianGreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig): 
    super().__init__(x, mu0, Sig0, Sig, False) 



warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(324)
tol = 1e-6


n_trials = 1
anms = ['GreedyKLReverse', 'GreedyKLForward', 'L1KLReverse', 'L1KLForward']
algs = [EGGreedyReverse, EGGreedyForward, EGL1Reverse, EGL1Forward]
algs_nms = list(zip(anms, algs))
tests = [(N, D, dist, algn) for N in [1, 10] for D in [1, 3] for dist in ['gauss', 'bin', 'axis_aligned'] for algn in algs_nms]


def gendata(N, D, dist="gauss"):
  if dist == "gauss":
    x = np.random.normal(0., 1., (N, D)) 
  elif dist == "bin":
    x = (np.random.rand(N, D) > 0.5).astype(float)
  else:
    D = N
    x = np.zeros((D, D))
    for i in range(D):
      x[i, i] = 1./float(D)

  mu0 = np.random.normal(0., 1., D)
  Sig0 = np.random.normal(0., 1., (D, D))
  Sig0 = Sig0.T.dot(Sig0)
  Sig = np.random.normal(0., 1., (D, D))
  Sig = Sig.T.dot(Sig)

  return x, mu0, Sig0, Sig

####################################################
#verifies that 
#-coreset size <= M at iteration M
#-error() vs output y(weights) are close to each other
#-error is decreasing
#-reset() resets the alg properly
#-build(M) with increasing M outputs same weights as one run with large M
#-if input size = 1, error is 0 for any M
####################################################
def coreset_single(N, D, dist, algn):
  #sys.stderr.write('n: ' + str(N) + ' d: ' +str(D) + ' dist: ' + str(dist) + ' salgn: ' + str(algn) + '\n')
  x, mu0, Sig0, Sig = gendata(N, D, dist)
  Sig0inv = np.linalg.inv(Sig0)
  Siginv = np.linalg.inv(Sig)
  mup, Sigp = weighted_post(mu0, np.linalg.inv(Sig0), np.linalg.inv(Sig), x, np.ones(x.shape[0]))
  anm, alg = algn
  coreset = alg(x, mu0, Sig0, Sig)

  #incremental M tests
  prev_err = np.inf
  for m in range(1, N+1):
    coreset.build(m)
    muw, Sigw = weighted_post(mu0, Sig0inv, Siginv, x, coreset.weights())
    w = coreset.weights()
    #check if coreset for 1 datapoint is immediately optimal
    if x.shape[0] == 1:
      assert np.fabs(w - np.array([1])) < tol, anm +" failed: coreset not immediately optimal with N = 1. weights: " + str(coreset.weights()) + " mup = " + str(mup) + " Sigp = " + str(Sigp) + " muw = " + str(muw) + " Sigw = " + str(Sigw) 
    #check if coreset is valid
    assert (w > 0.).sum() <= m, anm+" failed: coreset size > m"
    assert (w > 0.).sum() == coreset.size(), anm+" failed: sum of coreset.weights()>0  not equal to size(): sum = " + str((coreset.weights()>0).sum()) + " size(): " + str(coreset.size())
    assert np.all(w >= 0.), anm+" failed: coreset has negative weights"
    
 
    #check if actual output error is monotone
    err = weighted_post_KL(mu0, Sig0inv, Siginv, x, w, reverse=True if 'Reverse' in anm else False)
    assert err - prev_err < tol, anm+" failed: error is not monotone decreasing, err = " + str(err) + " prev_err = " +str(prev_err) 

    #check if coreset is computing error properly
    assert np.fabs(coreset.error() - err) < tol, anm+" failed: error est is not close to true err: est err = " + str(coreset.error()) + ' true err = ' + str(err)

    prev_err = err
  #save incremental M result
  w_inc = coreset.weights()

  #check reset
  coreset.reset()
  err = weighted_post_KL(mu0, Sig0inv, Siginv, x, np.zeros(x.shape[0]), reverse=True if 'Reverse' in anm else False)
  assert coreset.M == 0 and np.all(np.fabs(coreset.weights()) == 0.) and np.fabs(coreset.error() - err) < tol and not coreset.reached_numeric_limit, anm+" failed: reset() did not properly reset"

  #check build up to N all at once vs incremental
  #do this test for all except bin, where symmetries can cause instabilities in the choice of vector / weights
  if dist != 'bin':
    coreset.build(N)
    w = coreset.weights()
    err = weighted_post_KL(mu0, Sig0inv, Siginv, x, w, reverse=True if 'Reverse' in anm else False)
    err_inc = weighted_post_KL(mu0, Sig0inv, Siginv, x, w_inc, reverse=True if 'Reverse' in anm else False)
    assert np.sqrt(((w - w_inc)**2).sum()) < tol, anm+" failed: incremental buid up to N doesn't produce same result as one run at N : \n error = " +str(err) + " error_inc = " +  str(err_inc)
  #check if coreset with all_data_wts is optimal
  coreset._update_weights(coreset.all_data_wts)
  assert coreset.error() < tol, anm + " failed: coreset with all_data_wts does not have error 0"

def test_coreset():
  for N, D, dist, alg in tests:
    for n in range(n_trials):
      yield coreset_single, N, D, dist, alg

####################################################
#verifies that cst construction correctly responds to bad input
####################################################
    
def test_coreset_input_validation():
  for anm, alg in algs_nms:
    yield input_validation_single, alg, anm 

def input_validation_single(alg, anm):
  fe1 = False
  fe2 = False
  try:
    alg('fdas')
  except:
    fe1 = True
  try:
    alg(np.array(['fdsa', 'asdf']))
  except:
    fe2 = True

  if not fe1 or not fe2:
    assert False, anm + " failed: did not catch invalid input"

