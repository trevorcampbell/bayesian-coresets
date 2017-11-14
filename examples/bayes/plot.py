import numpy as np
import hilbertcoresets as hc
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import time

logFmtr = FuncTickFormatter(code="""
var trns = [
'\u2070',
'\u00B9',
'\u00B2',
'\u00B3',
'\u2074',
'\u2075',
'\u2076',
'\u2077',
'\u2078',
'\u2079'];
var tick_power = Math.floor(Math.log10(tick));
var tick_mult = Math.pow(10, Math.log10(tick) - tick_power);
var ret = '';
if (tick_mult > 1.) {
  if (Math.abs(tick_mult - Math.round(tick_mult)) > 0.05){
    ret = tick_mult.toFixed(1) + '\u22C5';
  } else {
    ret = tick_mult.toFixed(0) +'\u22C5';
  }
}
ret += '10';
if (tick_power < 0){
  ret += '\u207B';
}
ret += trns[Math.abs(tick_power)];
return ret;
""")

dnames = ['synth', 'ds1', 'phishing']
fldr = 'lr'


fig_ll = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Negative Test Log-Likelihood', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_w1 = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='1-Wasserstein', x_axis_label='M', plot_width=1250, plot_height=1250)

axis_font_size='30pt'
for f in [fig_ll, fig_w1]:
  #f.xaxis.ticker = bkm.tickers.FixedTicker(ticks=[.1, 1])
  f.xaxis.axis_label_text_font_size= axis_font_size
  f.xaxis.major_label_text_font_size= axis_font_size
  f.xaxis.formatter = logFmtr
  f.yaxis.axis_label_text_font_size= axis_font_size
  f.yaxis.major_label_text_font_size= axis_font_size
  f.yaxis.formatter = logFmtr
  f.toolbar.logo = None
  f.toolbar_location = None

pal = bokeh.palettes.colorblind['Colorblind'][len(dnames)]
for didx, dnm in enumerate(dnames):
  
  res = np.load(fldr +'/' + dnm  + '_results.npz')

  w1s = res['w1s']
  lls = res['lls']
  cputs = res['cputs']
  ll_fulls = res['ll_fulls']
  cput_fulls = res['cput_fulls']
  ll_max = res['ll_max']
  anms = res['anms']

  for aidx, anm in enumerate(anms):
    if anm == 'FW':
      ld = 'dashed'
    elif anm == 'RND':
      ld = 'dotted'
    else:
      ld = 'solid'
    #TODO: make this relative to full vs cput, and relative to RND vs M 
    fig_w1.line(np.percentile(cputs[aidx,:,:], 50, axis=0), np.percentile(w1s[aidx, :, :], 50, axis=0), line_color=pal[didx], line_width=4, line_dash=ld, legend=dnm if ld == 'solid' else None)
    fig_ll.line(np.percentile(cputs[aidx,:,:], 50, axis=0), np.percentile(ll_max - lls[aidx, :, :], 50, axis=0), line_color=pal[didx], line_width=4, line_dash=ld, legend=dnm if ld == 'solid' else None)
  
for f in [fig_ll, fig_w1]:
  f.legend.label_text_font_size= '16pt'
  f.legend.glyph_width=40
  f.legend.glyph_height=40
  f.legend.spacing=20

bkp.show(bkl.gridplot([[fig_ll, fig_w1]]))


