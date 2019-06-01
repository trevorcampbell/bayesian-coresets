import sys
import logging

#TODO: add a repeating handler for a log file,
#TODO and set repeat to false by default for the console
def add_handler(log, repeat=False, HandlerClass=logging.StreamHandler, handler_inits={'stream':sys.stderr}, format_string = '%(levelname)s - %(name)s.%(funcName)s(): %(message)s'):
  class CustomHandler(HandlerClass):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.prevmsgs = set()
    def emit(self, record):
      if not repeat: 
        if record.msg not in self.prevmsgs:
          self.prevmsgs.add(record.msg)
          super().emit(record)
      else: 
        super().emit(record)
  nrh = CustomHandler(**handler_inits)
  fmt = logging.Formatter(format_string)
  nrh.setFormatter(fmt)
  log.addHandler(nrh)

#TODO allow user to log to file
#%def set_logfile(fname):
#%  fh = logging.FileHandler(fname, mode='w')
#%  logging.basicConfig(stream=
  
