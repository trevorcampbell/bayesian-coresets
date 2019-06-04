import sys
import logging

#TODO: add a repeating handler for a log file, set default repeat to console = False, default repeat to log = True
def add_handler(log, repeat=False, HandlerClass=logging.StreamHandler, handler_inits={'stream':sys.stderr}, format_string = '%(levelname)s - %(name)s.%(funcName)s(): %(message)s'):
  class CustomHandler(HandlerClass):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.prevmsgs = set()
      self.n_suppressed = 0
    def emit(self, record):
      if not repeat: 
        if record.msg not in self.prevmsgs:
          self.prevmsgs.add(record.msg)
          super().emit(record)
        else:
          self.n_suppressed += 1 
      else: 
        super().emit(record)
  nrh = CustomHandler(**handler_inits)
  fmt = logging.Formatter(format_string)
  nrh.setFormatter(fmt)
  log.addHandler(nrh)

 
