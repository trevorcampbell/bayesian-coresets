import numpy as np
from bayesiancoresets import captree as ct

####
###This test is no longer used (tree-based search is not used in the main code)
####


np.random.seed(1)

n_trials = 10
tol = 1e-9
n_bound_samples = 100
tests = [(N, D, dist) for N in [10, 100] for D in [3, 10] for dist in ['gauss', 'bin', 'gauss_colinear', 'bin_colinear', 'axis_aligned']]

def gendata(N, D, dist="gauss"):
  if dist == "gauss":
    x = np.random.normal(0., 1., (N, D))
  elif dist == "bin":
    x = np.zeros((N, D))
    nz = (x**2).sum(axis=1) == 0.
    while nz.sum() > 0:
      x[nz, :] = (np.random.rand(nz.sum(), D) > 0.5).astype(float)
      nz = (x**2).sum(axis=1) == 0.
  elif dist == "gauss_colinear":
    x = np.random.normal(0., 1., D)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  elif dist == "bin_colinear":
    x = (np.random.rand(D) > 0.5).astype(float)
    while (x**2).sum() == 0:
      x = (np.random.rand(D) > 0.5).astype(float)
    y = np.random.rand(N)*2.-1.
    x = y[:, np.newaxis]*x
  else:
    x = np.zeros((N, N))
    for i in range(N):
      x[i, i] = 1./float(N)
  return x/np.sqrt((x**2).sum(axis=1))[:, np.newaxis]

####################################################
#verifies that the construction creates valid trees; checks
#-all data contained in xi.z >= r
#-there exists at least one vec = y
#-the index ny corresponds to the correct vector in x
#-there are 2*N-1 total nodes in the tree
####################################################
def check_tree_contents(node, x):
  if node.cR:
    xR, subtree_size_R = check_tree_contents(node.cR, x)
  if node.cL:
    xL, subtree_size_L = check_tree_contents(node.cL, x)
  assert (not node.cR and not node.cL) or (node.cR and node.cL), "cap tree validity test failed; cR xor cL is 1"
  if node.cR:
    nodex = np.vstack((xR, xL))
  else:
    nodex = np.atleast_2d(node.y)
    subtree_size_R = 0
    subtree_size_L = 0
  assert np.fabs(np.sqrt((node.xi**2).sum()) - 1.) < tol, "cap tree validity test failed; norm of xi is not 1"
  assert np.fabs(np.sqrt((node.y**2).sum()) - 1.) < tol, "cap tree validity test failed; norm of y is not 1"
  assert node.ny < x.shape[0] and node.ny >= 0, "cap tree validity test failed; ny index is not in [0, number of data points)"
  assert np.all(nodex.dot(node.xi) >= node.r-tol), "cap tree validity test failed; there is data that violates xi.y >= r"
  assert np.any(np.sqrt(((nodex - node.y)**2).sum(axis=1)) < tol), "cap tree validity test failed; y is not in the node data"
  assert np.sqrt(((x[node.ny, :] - node.y)**2).sum()) < tol, "cap tree validity test failed; the index ny is incorrect, x[node.ny, :] is not equal to node.y"
  return nodex, subtree_size_R+subtree_size_L+1
  
def tree_correctness_single(N, D, dist="gauss"):
  x = gendata(N, D, dist)
  root = ct.CapTree(x)
  _, tree_size = check_tree_contents(root, x)
  assert tree_size == 2*x.shape[0]-1, "cap tree validity test failed; tree is not a complete binary tree since # nodes is not 2*N-1"

def test_tree_correctness():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield tree_correctness_single, N, D, dist
  
####################################################
#verifies that the tree search produces correct results
#compared with linear search
#also verifies C tree
####################################################
def tree_search_single(N, D, dist="gauss"):
  x = gendata(N, D, dist)
  tree = ct.CapTree(x)
  for m in range(n_trials):
    yw = np.random.normal(0., 1., x.shape[1])
    yw /= np.sqrt((yw**2).sum())
    y_yw = np.random.normal(0., 1., x.shape[1])
    y_yw -= y_yw.dot(yw)*yw
    y_yw /= np.sqrt((y_yw**2).sum())
    n_ot = tree.search(yw, y_yw)
    f_ot = x[n_ot, :].dot(y_yw)/np.sqrt(1.-x[n_ot, :].dot(yw)**2)
    n_ol = (x.dot(y_yw)/np.sqrt(1.-x.dot(yw)**2)).argmax()
    f_ol = x[n_ol, :].dot(y_yw)/np.sqrt(1.-x[n_ol, :].dot(yw)**2)
    assert f_ol - f_ot < tol, "cap tree search failed; linear obj = " + str(f_ol) + " tree obj = " + str(f_ot)
      
def test_tree_search():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield tree_search_single, N, D, dist

####################################################
#verifies that the upper/lower bounds are valid
####################################################

def sample_within_r(xi, r, n):
  if xi.shape[0] == 1:
    if r > -1.:
      return xi[0]*np.ones(n)[:, np.newaxis]
    else:
      a = (np.random.rand(n) > 0.5).astype(float)
      a[a<0.5] = -1.
      return a[:, np.newaxis]
  z = np.random.normal(0., 1., (n, xi.shape[0]))
  z -= z.dot(xi)[:, np.newaxis]*xi
  z /= np.sqrt((z**2).sum(axis=1))[:, np.newaxis]
  th = np.arccos(r)*np.random.rand(n)[:, np.newaxis]
  return np.cos(th)*xi + np.sin(th)*z

def check_tree_bounds(node, yw, y_yw):
  if node.cR:
    check_tree_bounds(node.cR, yw, y_yw)
  if node.cL:
    check_tree_bounds(node.cL, yw, y_yw)
  v = sample_within_r(node.xi, node.r, n_bound_samples)
  assert v.shape == (n_bound_samples, node.xi.shape[0]), "v shape is wrong: vshape = " + str(v.shape) + " xishape = " + str(node.xi.shape)
  assert np.all(np.fabs((v**2).sum(axis=1) - 1.) < tol), "v is not normalized, norms = " + str((v**2).sum(axis=1))
  for n in range(yw.shape[0]):
    U = node.upper_bound(y_yw[n, :], yw[n, :])
    L = node.lower_bound(y_yw[n, :], yw[n, :])
    obj = v.dot(y_yw[n,:])/np.sqrt(1. - v.dot(yw[n,:])**2)
    assert np.all(obj - U < tol),  "cap tree bounds test failed; the node.upper_bound does not produce a valid upper bound"
    assert np.fabs(node.y.dot(y_yw[n,:])/np.sqrt(1. - node.y.dot(yw[n,:])**2) - L) < tol, "cap tree bounds test failed; the objective at node.y is not equal to node.lower_bound"

def tree_bounds_single(N, D, dist="gauss"):
  x = gendata(N, D, dist)
  root = ct.CapTree(x)
  yw = np.random.normal(0., 1., (n_bound_samples, x.shape[1]))
  yw /= np.sqrt((yw**2).sum(axis=1))[:, np.newaxis]
  y_yw = np.random.normal(0., 1., (n_bound_samples, x.shape[1]))
  y_yw -= (y_yw*yw).sum(axis=1)[:, np.newaxis]*yw
  y_yw /= np.sqrt((y_yw**2).sum(axis=1))[:, np.newaxis]
  check_tree_bounds(root, yw, y_yw)

def test_tree_bounds():
  for N, D, dist in tests:
    for n in range(n_trials):
      yield tree_bounds_single, N, D, dist


