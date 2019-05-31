from warnings import *


#monkey patch the warning format
def custom_warn(msg, *ar, **kw):
  return str(msg)+'\n'

formatwarning = custom_warn


