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
  tick_power = -tick_power;
}
power_digits = []
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 )
  tick_power = Math.floor(tick_power/10)
}
power_digits.push(tick_power)
for (i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")


#dnames = ['synth', 'ds1', 'phishing']
#fldr = 'lr'

dnames = ['synth', 'biketrips', 'airportdelays']
fldr = 'poiss'




fig_ll = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Negative Test Log-Likelihood', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_w1 = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='1-Wasserstein', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_kl = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Gaussian KL-Divergence', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)

axis_font_size='36pt'
legend_font_size='36pt'
for f in [fig_ll, fig_w1, fig_kl]:
  #f.xaxis.ticker = bkm.tickers.FixedTicker(ticks=[.1, 1])
  f.xaxis.axis_label_text_font_size= axis_font_size
  f.xaxis.major_label_text_font_size= axis_font_size
  f.xaxis.formatter = logFmtr
  f.yaxis.axis_label_text_font_size= axis_font_size
  f.yaxis.major_label_text_font_size= axis_font_size
  f.yaxis.formatter = logFmtr
  f.toolbar.logo = None
  f.toolbar_location = None

pal = bokeh.palettes.colorblind['Colorblind'][8]
pal = [pal[0], pal[1], pal[3], pal[4], pal[5], pal[6], pal[7], pal[2]]
for didx, dnm in enumerate(dnames):
  
  res = np.load(fldr +'/' + dnm  + '_results.npz')

  w1s = res['w1s']
  lls = res['lls']
  kls = res['kls']
  cputs = res['cputs']
  w1s_full = res['w1s_full']
  kls_full = res['kls_full']
  lls_full = res['lls_full']
  cputs_full = res['cputs_full']
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
    fig_w1.line(np.percentile(cputs[aidx,:,:], 50, axis=0), np.percentile(w1s[aidx, :, :], 50, axis=0), line_color=pal[didx], line_width=8, line_dash=ld, legend=dnm if ld == 'solid' else None)
    fig_ll.line(np.percentile(cputs[aidx,:,:], 50, axis=0), np.percentile(ll_max - lls[aidx, :, :], 50, axis=0), line_color=pal[didx], line_width=8, line_dash=ld, legend=dnm if ld == 'solid' else None)
    fig_kl.line(np.percentile(cputs[aidx,:,:], 50, axis=0), np.percentile(kls[aidx, :, :], 50, axis=0), line_color=pal[didx], line_width=8, line_dash=ld, legend=dnm if ld == 'solid' else None)
  
for f in [fig_ll, fig_w1, fig_kl]:
  f.legend.label_text_font_size= legend_font_size
  f.legend.glyph_width=40
  f.legend.glyph_height=40
  f.legend.spacing=20

bkp.show(bkl.gridplot([[fig_ll, fig_w1, fig_kl]]))


