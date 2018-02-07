import numpy as np
import heapq
from collections import deque

####
##This code is no longer used (linear search for giga works better)
####


class CapTree(object):
  def __init__(self, data):
    #if data is not None, this is the root, so set up the build queue and iterate construction
    self.nfun_node = 0.
    self.nfun_search = 0.
    if data is not None:
      self.nfun_construction = 0.
      build_queue = deque([])
      build_queue.append( (self, np.arange(data.shape[0])) )
      while build_queue:
        cap, idcs = build_queue.popleft()
        idcsR, idcsL, nf = cap.build(data, idcs)
        self.nfun_construction += nf
        if not cap.leaf:
          cap.cR = CapTree(None)
          cap.cL = CapTree(None)
          build_queue.append( (cap.cR, idcsR) )
          build_queue.append( (cap.cL, idcsL) )
  
  def is_build_done(self):
    return True

  def num_build_ops(self):
    return self.nfun_construction

  def num_search_ops(self):
    return self.nfun_search

  def num_search_nodes(self):
    return self.nfun_node

  def build(self, data, idcs):
    self.leaf = True
    self.cR = None
    self.cL = None
    if idcs.shape[0] == 1:
      self.y = data[idcs[0], :]
      self.xi = data[idcs[0], :]
      self.r = 1.
      self.ny = idcs[0]
      return None, None, 2. #nfun_constr
    else:
      self.leaf = False
      #compute manifold mean
      self.xi = data[idcs].sum(axis=0)
      xinrm = np.sqrt((self.xi**2).sum())
      #if xinrm is 0, just set it to the first datapoint
      if xinrm == 0.:
        self.xi = data[0, :]
      else:
        self.xi /= xinrm
      #get dists to all points
      dots = data[idcs].dot(self.xi)
      #get the closest point to the mean (for LB)
      nY = dots.argmax()
      self.y = data[idcs[nY], :]
      self.ny = idcs[nY]
      #get the furthest point (for r + children)
      nL = dots.argmin()
      self.r = max(-1., min(1., dots[nL])) #to take care of numerical > 1 or < -1
      #get the dists to the L anchor + furthest point
      dotsL = data[idcs].dot(data[idcs[nL], :])
      nR = dotsL.argmin()
      dotsR = data[idcs].dot(data[idcs[nR], :])
      #split based on L/R anchors
      idcsR = dotsR > dotsL
      #if all data are colinear, idcsR/idcsL can be empty, so just split in half
      if np.all(idcsR) or np.all(np.logical_not(idcsR)):
        idcsR = np.arange(idcs.shape[0]) < idcs.shape[0]/2
      #a better implementation of the above would do this in # O(d) operations:
      # if leaf
      #  2 operations to store xi and y
      # else
      #   N+2 for summing up data to get xi, normalizing, storing
      #   N+1 to compute argmin / argmax datapoint to xi + save argmax
      #   N to compute argmin (R child) from L child
      #   N to split based on dots from L and R
      nfun_constr = 4.*idcs.shape[0]+3.
      return idcs[idcsR], idcs[np.logical_not(idcsR)], nfun_constr
      

  def search(self, yw, y_yw):
    #each UB/LB computation is 2 O(d) operations
    pq = []
    L = -2.
    nopt = -1
    heapq.heappush(pq, (-self.upper_bound(y_yw, yw), self))
    nf = 2.
    nn = 1.
    while pq:
      negub, cap = heapq.heappop(pq)
      nn += 1.
      if -negub > L:
        ell = cap.lower_bound(y_yw, yw)
        nf += 2.
        if ell > L:
          L = ell
          nopt = cap.ny
        if not cap.leaf:
          uR = cap.cR.upper_bound(y_yw, yw)
          uL = cap.cL.upper_bound(y_yw, yw)
          if uR > L:
            heapq.heappush(pq, (-uR, cap.cR))
          if uL > L:
            heapq.heappush(pq, (-uL, cap.cL))
          nf += 4.
    self.nfun_search += nf
    self.nfun_node += nn
    return nopt

  def upper_bound(self, u, v):
    #compute upper bound
    bu = self.xi.dot(u)
    bv = self.xi.dot(v)
    b = np.sqrt(max(0., 1.-bu**2-bv**2))
    rv = np.sqrt(max(0., self.r**2 - bv**2))
    r1 = np.sqrt(max(0., 1.-self.r**2))
    if np.fabs(bv) > self.r or bu >= rv:
      return 1.
    else:
      return (bu*rv + b*r1)/(b**2+bu**2)

  def lower_bound(self, u, v):
    #compute lower bound
    bu = self.y.dot(u)
    bv = self.y.dot(v)
    if 1.-bv**2 <= 0. or bv <= -1.+1e-14: 
      #the first condition can occur when y = +/- y_w, and here the direction is not well defined 
      #the second can happen when y is roughly =  -y_w, and here the direction is not numerically stable
      #in either case, we want to return a failure - output = -3 indicates this
      return -3. 
    return bu/np.sqrt(1.-bv**2)


