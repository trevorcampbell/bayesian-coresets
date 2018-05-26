import bayesiancoresets as bc
import numpy as np
import warnings


warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-8

#linear test function/grad, sampling fcn for th and corresponding expectation gram matrix
def sample_linear(N, D):
  return np.random.randn(N, D)

def gram2_linear(x):
  return x.dot(x.T)

def gramF_linear(x):
  return x.dot(x.T)
  
def ll_linear(x, th):
  return x.dot(th[:,np.newaxis]).T

def gll_linear(x, th, idx=None):
  if idx is None:
    return x
  return x[:, idx]

#quadratic test function/grad, sampling fcn for th and corresponding expectation gram matrix
def sample_quad(N, D):
  return np.random.randn(N, D)

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
          grm[m,n] += 0.25*3*x[m,i]**2*x[n,i]
          continue
        #nunq == 2
        if (idcs == unq[0]).sum() == 3 or (idcs == unq[1]).sum() == 3:
          continue
        #2 groups of 2
        grm[m,n] += 0.25*x[m,i]*x[n,j]*x[m,k]*x[n,l]  
  return grm

def gramF_quad(x):
  return (x.dot(x.T))**2
  
def ll_quad(x, th):
  return 0.5*(x.dot(th[:,np.newaxis]).T)**2

def gll_quad(x, th, idx=None):
  if idx is None:
    return x.dot(th[:,np.newaxis])*x
  return x.dot(th[:,np.newaxis]).T * x[:,idx]

#evaluate the testing ll / gll functions to make sure there's no error in the tests themselves
def single_llgll(ll, gll):
  th = np.random.randn(5)
  x = np.random.randn(10, 5)
  #compute exact gradient
  exact_grad = gll(x, th)
  #make sure it has the right shape and the component evals are equal
  assert exact_grad.shape == (10, 5), "error: grad has incorrect shape"
  for i in range(5):
    assert np.all(exact_grad[:,i] == gll(x, th, i)), "error: grad().component != grad(component)"
  #compare the numerical grad
  num_grad = np.zeros((10, 5))
  for i in range(5):
    thr = th.copy()
    thr[i] += tol
    thl = th.copy()
    thl[i] -= tol
    num_grad[:, i] = (ll(x, thr) - ll(x, thl))/(2*tol)
  assert np.all(np.fabs(num_grad - exact_grad) < 100*tol), "error: numerical/exact gradients do not match up; max diff = " + str(np.fabs(num_grad-exact_grad).max())

def test_llgll():
  for ll, gll in [(ll_linear, gll_linear), (ll_quad, gll_quad)]:
    yield single_llgll, ll, gll
  
#test if the F projection converges to the expectation
def test_projF():
  bc.Projection2(data, log_likelihood, projection_dim, sample_approx_posterior)
  bc.ProjectionF(data, grad_log_likelihood, projection_dim, sample_approx_posterior)

#test if 2 projection converges to its expectation
def test_proj2():
  
#test F on gaussian data, ensure converges to mean
#def test_projection_gaussian_data():
#  pass #TODO

#pass in garbage and make sure it catches it
#def test_projection_input_validation():
#  pass #TODO
 

