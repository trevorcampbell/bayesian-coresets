import sys
import logging


LOGLEVELS = {'error': logging.ERROR, 'warning':logging.WARNING, 'critical':logging.CRITICAL, 'info':logging.INFO, 'debug':logging.DEBUG, 'notset':logging.NOTSET}
def set_verbosity(verb):
  logging.getLogger().setLevel(LOGLEVELS[verb])

#def set_repeat(repeat):
#  logging.getLogger().handlers[0].repeat_flag = repeat

#TODO: add a repeating handler for a log file, set default repeat to console = False, default repeat to log = True
def add_handler(log, repeat_flag, HandlerClass=logging.StreamHandler, handler_inits={'stream':sys.stderr}, format_string = '%(levelname)s - %(id)s.%(funcName)s(): %(message)s'):
  class CustomHandler(HandlerClass):
    pass
    #def __init__(self, *args, **kwargs):
    #  super().__init__(*args, **kwargs)
    #  self.prevmsgs = {}
    #  self.repeat_flag = False

    #def emit(self, record):
    #  if not self.repeat_flag:
    #    if record.msg not in self.prevmsgs.keys():
    #      self.prevmsgs[record.msg] = 0
    #      super().emit(record)
    #    else:
    #      self.prevmsgs[record.msg] += 1
    #  else: 
    #    super().emit(record)

    #def remove_all(self, nm):
    #  n_removed = sum([self.prevmsgs[key] for key in self.prevmsgs.keys() if nm in key])
    #  self.prevmsgs = {key : self.prevmsgs[key] for key in self.prevmsgs.keys() if nm not in key}
    #  return n_removed
   
  nrh = CustomHandler(**handler_inits)
  fmt = logging.Formatter(format_string)
  nrh.setFormatter(fmt)
  log.addHandler(nrh)

logging.getLogger().setLevel(LOGLEVELS['error'])
add_handler(logging.getLogger(), False)
