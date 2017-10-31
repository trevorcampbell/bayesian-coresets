import hilbertcoresets as hc
import numpy as np

n_trials = 50
tol = 1e-16

####################################################
#verifies that FW works under normal operating conditions
#-coreset size <= M at iteration M
#-error() vs output y(weights) are close to each other
#-error is decreasing
#-reset() resets the alg properly
#-run(M) with increasing M outputs same weights as 
# one run with large M
####################################################
def fw_normal_operation_single(N, D, dist="gauss"):
  for n in range(n_trials):
    if dist == "gauss":
      x = np.random.normal(0., 1., (N, D))
    else:
      x = (np.random.rand(N, D) > 0.5).astype(int)
    xs = x.sum(axis=0)
    fw = hc.FrankWolfe(x)
    #incremental M tests
    prev_err = np.sqrt(((x - xs)**2).sum(axis=1)).max()
    for m in range(1, N+1):
      fw.run(m)
      assert (fw.weights() > 0.).sum() <= m, "fw normal ops failed: coreset size > m"
      xw = (fw.weights()[:, np.newaxis]*x).sum(axis=0)
      assert np.sqrt(((xw-xs)**2).sum()) < prev_err, "fw normal ops failed: error is not monotone"
      assert np.fabs(fw.error() - np.sqrt(((xw-xs)**2).sum())) < tol, "fw normal ops failed: x(w) est is not close to true x(w)"
      prev_err = np.sqrt(((xw-xs)**2).sum())
    #save incremental M result
    w_inc = fw.weights()
    xw_inc = (fw.weights()[:, np.newaxis]*x).sum(axis=0) 
    
    #check reset
    fw.reset()
    assert fw.M == 0 and np.all(np.fabs(fw.weights()) < tol) and np.fabs(fw.error() - np.sqrt((xs**2).sum())) < tol, "fw normal ops failed: fw.reset() did not properly reset"
    #check reset
    fw.run(N)
    xw = (fw.weights()[:, np.newaxis]*x).sum(axis=0) 
    assert np.all(np.fabs(fw.weights() - w_inc) < tol) and np.sqrt(((xw-xw_inc)**2).sum()) < tol, "fw normal ops failed: incremental run up to N doesn't produce same result as one run at N"

def test_fw_normal_operation():
  tests = [(N, D, dist) for N in [0, 1, 1000] for D in [0, 1, 10] for dist in ['gauss', 'bin']]
  for N, D, dist in tests:
    yield fw_normal_operation_single(N, D, dist)
 
####################################################
#verifies FW on input size 1 gets error 0 immediately
####################################################
def test_fw_single_input():
  pass #TODO
 
 
####################################################
#verifies FW on colinear data gets error 0 immediately
####################################################
def test_fw_colinear():
  pass #TODO
 
 
####################################################
#verifies FW on axis-aligned data has error = known formula
####################################################
def test_fw_axis_aligned():
  pass #TODO
 

####################################################
#verifies the sqrt upper bound for FW is actually an upper bound
####################################################
def test_fw_sqrt_bound():
  pass #TODO
 

####################################################
#verifies the geometric upper bound for FW is actually an upper bound
####################################################
def test_fw_geometric_bound():
  pass #TODO

####################################################
#verifies that FW correctly responds to bad input
####################################################
   
def test_tree_input_validation():
  #empty
  #non array
  #non number
  #integer vs real
  #m = 0
  pass #TODO
 
  
 

