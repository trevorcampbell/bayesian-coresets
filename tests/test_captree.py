import hilbertcoresets as hc
import numpy as np

#tests:
##construction: walk up the tree and make sure all data is actually within bounds and exists a y at index with lb cost
##construction: corner cases: empty, one, data is 1d
##search: make sure it lines up with linear for randomly selected
##search: make sure it lines up with linear if one of the data = the query point
##search: corner cases
