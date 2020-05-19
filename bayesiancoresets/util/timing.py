import time

__tt = 0
def _tic():
    __tt = time.perf_counter()

def _toc():
  return time.perf_counter() - __tt
