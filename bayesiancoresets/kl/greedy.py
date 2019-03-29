class GreedyKLCoreset(KLCoreset,GreedySingleUpdateCoreset):

  def _search(self):
    raise NotImplementedError()

  def _step_coeffs(self, f):
    raise NotImplementedError()

  def _prepare_retry_step(self):
    pass #implementation optional
  
  def _prepare_retry_search(self):
    pass #implementation optional

  def _update_cache(self, alpha, beta, f):
    pass #implementation optional

  


