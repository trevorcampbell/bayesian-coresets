import bayesiancoresets as bc
import autograd.numpy as np
import warnings

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
  def __init__(self, x, mu0, Sig0, Sig):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(None, None, 0)

  def _compute_scales(self):
    return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl_estimate(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=False)

  def _reverse_kl_estimate(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=True)

  def _forward_kl_grad_estimate(self, w, normalize):
    g = lambda w : grad(weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=False))
    return g(w)

  def _reverse_kl_grad_estimate(self, w, normalize):
    g = lambda w : grad(weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=True))
    if normalize:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)

class ExactGaussianGreedyKLCoreset(bc.GreedyKLCoreset):
  def __init__(self, x, mu0, Sig0, Sig):
    self.x = x
    self.mu0 = mu0
    self.Sig0 = Sig0
    self.Sig0inv = np.linalg.inv(Sig0)
    self.Sig = Sig
    self.Siginv = np.linalg.inv(Sig)
    super().__init__(None, None, 0)

  def _compute_scales(self):
    return np.sqrt(ll_m2_exact_diag(self.mu0, self.Sig0, self.Siginv, self.x))

  def _forward_kl_estimate(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=False)

  def _reverse_kl_estimate(self):
    return weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, self.wts, reverse=True)

  def _forward_kl_grad_estimate(self, w, normalize):
    g = lambda w : grad(weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=False))
    return g(w)

  def _reverse_kl_grad_estimate(self, w, normalize):
    g = lambda w : grad(weighted_post_KL(self.mu0, self.Sig0inv, self.Siginv, self.x, w, reverse=True))
    if normalize:
      muw, Sigw = weighted_post(self.mu0, self.Sig0inv, self.Siginv, self.x, w)
      return g(w)/np.sqrt(ll_m2_exact_diag(muw, Sigw, self.Siginv, self.x))
    else:
      return g(w)


warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(324)
tol = 1e-6


n_trials = 10
anms = ['GreedyKL', 'L1KL']
algs = [ExactGaussianGreedyKLCoreset, ExactGaussianL1KLCoreset]
algs_nms = list(zip(anms, algs))
tests = [(N, D, dist, algn) for N in [1, 10, 100] for D in [1, 3, 10] for dist in ['gauss', 'bin', 'gauss_colinear', 'bin_colinear', 'axis_aligned'] for algn in algs_nms]


def gendata(N, D, dist="gauss"):
  if dist == "gauss":
    x = np.random.normal(0., 1., (N, D))
  elif dist == "bin":
    x = (np.random.rand(N, D) > 0.5).astype(float)
  elif dist == "gauss_colinear":
    x = np.random.normal(0., 1., D)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  elif dist == "bin_colinear":
    x = (np.random.rand(D) > 0.5).astype(float)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  else:
    x = np.zeros((N, N))
    for i in range(N):
      x[i, i] = 1./float(N)
  return x

####################################################
#verifies that 
#-coreset size <= M at iteration M
#-error() vs output y(weights) are close to each other
#-error is decreasing
#-reset() resets the alg properly
#-build(M) with increasing M outputs same weights as one run with large M
#-if input size = 1, error is 0 for any M
#-if input is colinear, error is 0 forall M
####################################################
def coreset_single(N, D, dist, algn):
  x = gendata(N, D, dist)
  xs = x.sum(axis=0)
  anm, alg = algn
  coreset = alg(x, use_cached_xw=True)
  accuratecoreset = alg(x, use_cached_xw=False)

  #incremental M tests
  prev_err = np.inf
  for m in range(1, N+1):
    coreset.build(m)
    accuratecoreset.build(m)
    #check if coreset for 1 datapoint is immediately optimal
    if x.shape[0] == 1:
      assert np.fabs(coreset.weights(optimal_scaling=True) - np.array([1])) < tol or (np.fabs(coreset.weights(optimal_scaling=True) - np.array([0])) < tol and (x**2).sum() == 0.), anm +" failed: coreset not immediately optimal with N = 1. weights: " + str(coreset.weights(optimal_scaling=True))
    #check if coreset is valid
    assert (coreset.weights() > 0.).sum() <= m, anm+" failed: coreset size > m"
    assert (coreset.weights() > 0.).sum() == coreset.size(), anm+" failed: sum of coreset.weights()>0  not equal to size(): sum = " + str((coreset.weights()>0).sum()) + " size(): " + str(coreset.size())
    assert np.all(coreset.weights() >= 0.), anm+" failed: coreset has negative weights"
    
    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0)
    xwopt = (coreset.weights(optimal_scaling=True)[:, np.newaxis]*x).sum(axis=0)
 
    #check if actual output error is monotone
    assert np.sqrt(((xw-xs)**2).sum()) - prev_err < tol, anm+" failed: error is not monotone decreasing, err = " + str(np.sqrt(((xw-xs)**2).sum())) + " prev_err = " +str(prev_err) 

    #check if coreset is computing error properly
    #without optimal scaling
    assert np.fabs(coreset.error() - np.sqrt(((xw-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w): est err = " + str(coreset.error()) + ' true err = ' + str(np.sqrt(((xw-xs)**2).sum()))
    #with optimal scaling
    assert np.fabs(coreset.error(optimal_scaling=True) - np.sqrt(((xwopt-xs)**2).sum())) < tol, anm+" failed: x(w) est is not close to true x(w) with optimal scaling: est err = " + str(coreset.error(optimal_scaling=True)) + ' true err = ' + str(np.sqrt(((xwopt-xs)**2).sum()))

    #check if fast / accurate error estimates are close
    #without optimal scaling
    assert np.fabs(coreset.error() - accuratecoreset.error()) < tol*1000, anm+" failed: error(accurate/fast) do not return similar results: fast err = " + str(coreset.error()) + ' acc err = ' + str(accuratecoreset.error())
    #with optimal scaling
    assert np.fabs(accuratecoreset.error(optimal_scaling=True) - coreset.error(optimal_scaling=True)) < tol*1000, anm+" failed: error(accurate/fast) with optimal scaling do not return similar results: fast err = " + str(coreset.error(optimal_scaling=True)) + ' acc err = ' + str(accuratecoreset.error(optimal_scaling=True))

    #ensure optimally scaled error is lower than  regular
    assert coreset.error(optimal_scaling=True) - coreset.error() < tol, anm+" failed: optimal scaled coreset produces higher acc error than regular one. Optimal err = " + str(coreset.error(optimal_scaling=True)) + ' regular err: ' + str(coreset.error())

    #if data are colinear, check if the coreset is optimal immediately
    if 'colinear' in dist and m >= 1:
      assert np.sqrt(((xwopt-xs)**2).sum()) < tol, anm + " failed: colinear data, m>= 1 not immediately optimal:  optimal scaled err " + str(np.sqrt(((xwopt-xs)**2).sum())) + " tol = " + str(tol) + " m = " + str(m) + ' xwopt = ' + str(xwopt) + ' xs = ' + str(xs)
    ##if data are axis aligned, 
    #if 'axis' in dist:
    #  assert np.all( np.fabs(coreset.weights()[ coreset.weights() > 0. ] - 1. ) < tol ), anm+" failed: on axis-aligned data, weights are not 1"
    #  assert np.fabs(np.sqrt(((xw-xs)**2).sum())/np.sqrt((xs**2).sum()) - np.sqrt(1. - float(m)/float(N))) < tol, anm+" failed: on axis-aligned data, error is not sqrt(1 - M/N)"
    prev_err = np.sqrt(((xw-xs)**2).sum())
    prev_wts = coreset.wts.copy()
  #save incremental M result
  w_inc = coreset.weights()
  xw_inc = (coreset.weights()[:, np.newaxis]*x).sum(axis=0) 

  #check reset
  coreset.reset()
  assert coreset.M == 0 and np.all(np.fabs(coreset.weights()) == 0.) and np.fabs(coreset.error() - np.sqrt((xs**2).sum())) < tol and not coreset.reached_numeric_limit, anm+" failed: reset() did not properly reset"

  #check build up to N all at once vs incremental
  #do this test for all except bin, where symmetries can cause instabilities in the choice of vector / weights
  if dist != 'bin':
    coreset.build(N)
    xw = (coreset.weights()[:, np.newaxis]*x).sum(axis=0) 
    assert np.sqrt(((xw-xw_inc)**2).sum()) < tol, anm+" failed: incremental buid up to N doesn't produce same result as one run at N : \n xw = " + str(xw) + " error = " +str(np.sqrt(((xw-xs)**2).sum())) + " \n xw_inc = " + str(xw_inc) + " error = " +  str(np.sqrt(((xw_inc-xs)**2).sum())) + " \n xs = " +str(xs)


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

