import sys
import logging


LOGLEVELS = {'error': logging.ERROR, 'warning':logging.WARNING, 'critical':logging.CRITICAL, 'info':logging.INFO, 'debug':logging.DEBUG, 'notset':logging.NOTSET}
def set_verbosity(verb):
  logging.getLogger().setLevel(LOGLEVELS[verb])

def set_repeat(repeat):
  logging.getLogger().handlers[0].repeat_flag = repeat

#TODO: add a repeating handler for a log file, set default repeat to console = False, default repeat to log = True
def add_handler(log, repeat_flag, HandlerClass=logging.StreamHandler, handler_inits={'stream':sys.stderr}, format_string = '%(levelname)s - %(id)s.%(funcName)s(): %(message)s'):
  class CustomHandler(HandlerClass):
    def __init__(self, *args, **kwargs):
      super().__init__(*args, **kwargs)
      self.prevmsgs = set()
      self.n_suppressed = 0
      self.repeat_flag = False
    def emit(self, record):
        if not self.repeat_flag:
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

logging.getLogger().setLevel(LOGLEVELS['error'])
add_handler(logging.getLogger(), False)
