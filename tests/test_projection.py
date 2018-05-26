import bayesiancoresets as bc
import numpy as np
import warnings


warnings.filterwarnings('ignore', category=UserWarning) #tests will generate warnings (due to pathological data design for testing), just ignore them
np.seterr(all='raise')
np.set_printoptions(linewidth=500)
np.random.seed(100)
tol = 1e-9

#linear test function/grad
def ll_linear(x, th):
  return x.dot(th[:,np.newaxis])
def gll_linear(x, th, idx=None):
  if idx is None:
    return x
  return x[:, idx]

#quadratic test function/grad
def ll_quad(x, th):
  return x.dot((th[:,np.newaxis]*th).dot(x.T))

def gll_quad(x, th, idx):
  if idx is None:
    return 2*x.dot(th[:,np.newaxis])*x
  return 2*x.dot(th[:,np.newaxis]).T * x[:,idx]

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
  assert np.all(np.fabs(num_grad - exact_grad) < tol), "error: numerical/exact gradients do not match up; diff = " + str(np.fabs(num_grad-exact_grad).sum())

def test_llgll():
  for ll, gll in [(ll_linear, gll_linear), (ll_quad, gll_quad)]:
    yield single_llgll, ll, gll
  
#def test_projF():
#  bc.Projection2(data, log_likelihood, projection_dim, sample_approx_posterior)
#  bc.ProjectionF(data, grad_log_likelihood, projection_dim, sample_approx_posterior)

#test F on gaussian data, ensure converges to mean
#def test_projection_gaussian_data():
#  pass #TODO

#pass in garbage and make sure it catches it
#def test_projection_input_validation():
#  pass #TODO
 

