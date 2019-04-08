import bayesiancoresets as bc
import numpy as np
import warnings


warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
Nsamps=10000000

#linear test function/grad, sampling fcn for th and corresponding expectation gram matrix
def sample_linear(N, D):
  return np.random.randn(N, D)

def ll_linear(x, th):
  return (x.dot(th[:,np.newaxis])).flatten()

def gll_linear(x, th, idx=None):
  if idx is None:
    return x
  return x[:, idx]

def gram2_linear(x):
  return x.dot(x.T)

def gramF_linear(x):
  return x.dot(x.T)
  
#quadratic test function/grad, sampling fcn for th and corresponding expectation gram matrix
def sample_quad(N, D):
  return np.random.randn(N, D)

def ll_quad(x, th):
  return (0.5*(x.dot(th[:,np.newaxis]))**2).flatten()

def gll_quad(x, th, idx=None):
  if idx is None:
    return x.dot(th[:,np.newaxis])*x
  return x.dot(th[:,np.newaxis]).T * x[:,idx]

def gram2_quad(x):
  #init gram matrix
  grm = np.zeros((x.shape[0], x.shape[0]))
  irng = range(x.shape[1])
  idxqs = [(i,j,k,l) for i in irng for j in irng for k in irng for l in irng]
  #loop over all pairs of data
  for m in range(x.shape[0]):
    for n in range(x.shape[0]):
      #loop over all index quartets
      for i, j, k, l in idxqs:
        idcs = np.array([i,j,k,l])
        unq = np.unique(idcs)
        nunq = unq.shape[0]
        if nunq == 3 or nunq == 4:
          continue
        if nunq == 1:
          grm[m,n] += 0.25*3*x[m,i]**2*x[n,i]**2
          continue
        #nunq == 2
        if (idcs == unq[0]).sum() == 3 or (idcs == unq[1]).sum() == 3:
          continue
        #2 groups of 2
        grm[m,n] += 0.25*x[m,i]*x[n,j]*x[m,k]*x[n,l]  
  return grm

def gramF_quad(x):
  return (x.dot(x.T))**2
  
#evaluate the testing ll / gll functions to make sure there's no error in the tests themselves
def single_llgll(ll, gll, g2, gF, samp):
  th = np.random.randn(2)
  x = np.random.randn(3, 2)
  #compute exact gradient
  exact_grad = gll(x, th)
  #make sure it has the right shape and the component evals are equal
  assert exact_grad.shape == (3, 2), "error: grad has incorrect shape"
  for i in range(2):
    assert np.all(exact_grad[:,i] == gll(x, th, i)), "error: grad().component != grad(component)"
  #compare the numerical grad
  num_grad = np.zeros((3,2))
  eps = 1e-9
  for i in range(2):
    thr = th.copy()
    thr[i] += eps
    thl = th.copy()
    thl[i] -= eps
    num_grad[:, i] = (ll(x, thr) - ll(x, thl))/(2*eps)
  assert np.all(np.fabs(num_grad - exact_grad) < 1e-6), "error: numerical/exact gradients do not match up; max diff = " + str(np.fabs(num_grad-exact_grad).max())
  #make sure the exact expected gram matrices are close to numerical values
  exact_gram2 = g2(x)
  exact_gramF = gF(x)
  ths = samp(Nsamps, 2)
  num_gram2 = np.zeros_like(exact_gram2)
  num_gramF = np.zeros_like(exact_gramF)
  for i in range(Nsamps):
    lls = ll(x, ths[i, :])
    glls = gll(x, ths[i, :])
    num_gram2 += lls[:, np.newaxis]*lls
    num_gramF += glls.dot(glls.T)
  num_gram2 /= Nsamps
  num_gramF /= Nsamps
  assert np.all(np.fabs(num_gramF - exact_gramF) < 5e-2), "error: numerical/exact gramF matrices don't match up; max diff = " + str(np.fabs(num_gramF-exact_gramF).max())
  assert np.all(np.fabs(num_gram2 - exact_gram2) < 5e-2), "error: numerical/exact gram2 matrices don't match up; max diff = " + str(np.fabs(num_gram2-exact_gram2).max())
  
def test_llgll():
  for ll, gll, g2, gF, samp in [(ll_linear, gll_linear, gram2_linear, gramF_linear, sample_linear), (ll_quad, gll_quad, gram2_quad, gramF_quad, sample_quad)]:
    yield single_llgll, ll, gll, g2, gF, samp
  
#test if the F projection converges to the expectation
def single_projF(gll, gram, samp):
  x = samp(3, 2)
  proj = bc.ProjectionF(x, gll, Nsamps, lambda : samp(1, 2).flatten())
  w = proj.get()
  assert np.all(np.fabs(gram(x) - w.dot(w.T)) < 1e-2), "error: projectionF doesn't converge to expectation; max diff = " + str(np.fabs(gram(x) - w.dot(w.T)).max())
  proj.reset()
  assert proj.get().shape == w.shape, "error: proj.reset() doesn't retain shape"

  is_constant = True
  gtest = gll(x, samp(1, 2).flatten())
  for i in range(10):
    gtest2 = gll(x, samp(1,2).flatten())
    if np.any(gtest2 != gtest):
      is_constant = False
  if not is_constant:
    assert np.all(np.fabs(w - proj.get()) > 0), "error: proj.reset() doesn't refresh entries"

  proj.reset(5)
  assert proj.get().shape[1] == 5, "error: proj.reset(5) doesn't create a new projection with 5 components"

#test if 2 projection converges to its expectation
def single_proj2(ll, gram, samp):
  x = samp(3, 2)
  proj = bc.Projection2(x, ll, Nsamps, lambda : samp(1, 2).flatten())
  w = proj.get()
  assert np.all(np.fabs(gram(x) - w.dot(w.T)) < 1e-2), "error: projection2 doesn't converge to expectation; max diff = " + str(np.fabs(gram(x) - w.dot(w.T)).max())
  proj.reset()
  assert proj.get().shape == w.shape, "error: proj.reset() doesn't retain shape"
 
  is_constant = True
  ltest = ll(x, samp(1, 2).flatten())
  for i in range(10):
    ltest2 = ll(x, samp(1,2).flatten())
    if np.any(ltest2 != ltest):
      is_constant = False
  if not is_constant:
    assert np.all(np.fabs(w - proj.get()) > 0), "error: proj.reset() doesn't refresh entries"

  proj.reset(5)
  assert proj.get().shape[1] == 5, "error: proj.reset(5) doesn't create a new projection with 5 components"

def test_proj():
  for ll, gll, g2, gF, samp in [(ll_linear, gll_linear, gram2_linear, gramF_linear, sample_linear), (ll_quad, gll_quad, gram2_quad, gramF_quad, sample_quad)]:
    yield single_projF, gll, gF, samp
    yield single_proj2, ll, g2, samp

