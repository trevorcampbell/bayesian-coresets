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


fig_err_g = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Err', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_cost_g = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Time (ms) & #Ops', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_err_a = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Err', x_axis_label='M', plot_width=1250, plot_height=1250)
fig_cost_a = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Time (ms) & #Ops', x_axis_label='M', plot_width=1250, plot_height=1250)

axis_font_size='30pt'
for f in [fig_err_g, fig_cost_g, fig_err_a, fig_cost_a]:
  #f.xaxis.ticker = bkm.tickers.FixedTicker(ticks=[.1, 1])
  f.xaxis.axis_label_text_font_size= axis_font_size
  f.xaxis.major_label_text_font_size= axis_font_size
  f.xaxis.formatter = logFmtr
  f.yaxis.axis_label_text_font_size= axis_font_size
  f.yaxis.major_label_text_font_size= axis_font_size
  f.yaxis.formatter = logFmtr
  f.toolbar.logo = None
  f.toolbar_location = None


gr = np.load('gauss_results.npz')
anms = gr['anms']
Ms = gr['Ms']
err = gr['err']
nfunc = gr['nfunc']
cput = gr['cput']
pal = bokeh.palettes.colorblind['Colorblind'][len(anms)]
for aidx, anm in enumerate(anms):
  fig_err_g.line(Ms, np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=4, legend=anm)
  fig_cost_g.line(Ms, np.percentile(nfunc[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=4, legend=anm)
  fig_cost_g.line(Ms, np.percentile(cput[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=4, line_dash='dashed')


aa = np.load('axis_results.npz')
anms = aa['anms']
Ms = aa['Ms']
err = aa['err']
nfunc = aa['nfunc']
cput = aa['cput']
pal = bokeh.palettes.colorblind['Colorblind'][len(anms)]
for aidx, anm in enumerate(anms):
  fig_err_a.line(Ms, np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=4, legend=anm)
  fig_cost_a.line(Ms, np.percentile(nfunc[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=4, legend=anm)
  fig_cost_a.line(Ms, np.percentile(cput[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=4, line_dash='dashed')

 
for f in [fig_err_g, fig_cost_g, fig_err_a, fig_cost_a]:
  f.legend.label_text_font_size= '16pt'
  f.legend.glyph_width=40
  f.legend.glyph_height=40
  f.legend.spacing=20

bkp.show(bkl.gridplot([[fig_err_g, fig_cost_g], [fig_err_a, fig_cost_a]]))


