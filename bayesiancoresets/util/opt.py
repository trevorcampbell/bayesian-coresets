import numpy as np
import sys

def nn_opt(x0, grd, opt_itrs=1000, step_sched = lambda i : 1./(i+1), b1=0.9, b2=0.99, eps=1e-8, verbose=False):
  x = x0.copy()
  m1 = np.zeros(x.shape[0])
  m2 = np.zeros(x.shape[0])
  for i in range(opt_itrs):
    g = grd(x)
    if verbose:
      sys.stdout.write('itr ' + str(i+1) +'/'+str(opt_itrs)+': ||inactive constraint grads|| = ' + str(np.sqrt((g[x>0]**2).sum())) + '                \r')
      sys.stdout.flush()
    m1 = b1*m1 + (1.-b1)*g
    m2 = b2*m2 + (1.-b2)*g**2
    upd = step_sched(i)*m1/(1.-b1**(i+1))/(eps + np.sqrt(m2/(1.-b2**(i+1))))
    x -= upd

    #project onto x>=0
    x = np.maximum(x, 0.)

  if verbose:
    sys.stdout.write('\n')
    sys.stdout.flush()

  return x





