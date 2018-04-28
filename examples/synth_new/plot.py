import numpy as np
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
if (tick <= 0){
  return '';
}
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
if (tick_power < 0) {
  ret += '\u207B';
  tick_power = tick_power*(-1);
}
power_digits = [];
while (tick_power > 9){
  power_digits.push( tick_power - Math.floor(tick_power/10)*10 );
  tick_power = Math.floor(tick_power/10);
}

power_digits.push(tick_power);
for (i = power_digits.length-1; i >= 0; i--){
  ret += trns[power_digits[i]];
}
return ret;
""")


fig_err_g = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Error', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_csz_g = bkp.figure(x_axis_type='log', y_axis_type='log', y_axis_label='Coreset Size', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_t_g = bkp.figure(x_axis_type='log', y_axis_type='log', y_axis_label='CPU Time (s)', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_errc_g = bkp.figure(x_axis_type='log', y_axis_type='log', y_axis_label='Error', x_axis_label='Coreset Size', plot_width=1250, plot_height=1250)
fig_errt_g = bkp.figure(x_axis_type='log', y_axis_type='log', y_axis_label='Error', x_axis_label='CPU Time (s)', plot_width=1250, plot_height=1250)
fig_err_a = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Error', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_csz_a = bkp.figure(x_axis_type='log',y_axis_type='log', y_axis_label='Coreset Size', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_t_a = bkp.figure(x_axis_type='log', y_axis_type='log', y_axis_label='CPU Time (s)', x_axis_label='Coreset Construction Iterations', plot_width=1250, plot_height=1250)
fig_errc_a = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Error', x_axis_label='Coreset Size', plot_width=1250, plot_height=1250)
fig_errt_a = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Error', x_axis_label='CPU Time (s)', plot_width=1250, plot_height=1250)

axis_font_size='36pt'
legend_font_size='36pt'
for f in [fig_err_g, fig_err_a, fig_csz_a, fig_csz_g, fig_t_a, fig_t_g, fig_errc_g, fig_errc_a, fig_errt_g, fig_errt_a]:
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
cput = gr['cput']
csize = gr['csize']
pal = bokeh.palettes.colorblind['Colorblind'][8]
pal = [pal[0], pal[1], '#d62728', pal[4], pal[6], pal[3], pal[7], pal[2]]
for aidx, anm in enumerate(anms):
  fig_err_g.line(Ms, np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'))
  fig_err_g.line(Ms, np.percentile(scaled_err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, line_dash='dotted')
  fig_csz_g.line(Ms, np.percentile(csize[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'))
  fig_t_g.line(Ms, np.percentile(cput[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'))
  fig_errc_g.line(np.percentile(csize[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'))
  fig_errc_g.line(np.percentile(csize[aidx,:,:], 50, axis=0), np.percentile(scaled_err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, line_dash='dotted')
  fig_errt_g.line(np.percentile(cput[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'))
  fig_errt_g.line(np.percentile(cput[aidx,:,:], 50, axis=0), np.percentile(scaled_err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, line_dash='dotted')


aa = np.load('axis_results.npz')
anms = aa['anms']
Ms = aa['Ms']
err = aa['err']
cput = aa['cput']
csize = aa['csize']
#pal = bokeh.palettes.colorblind['Colorblind'][len(anms)]
for aidx, anm in enumerate(anms):
  fig_err_a.line(Ms, np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'), line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_err_a.line(Ms, np.percentile(scaled_err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, line_dash='dotted')
  fig_csz_a.line(Ms, np.percentile(csize[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'), line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_t_a.line(Ms, np.percentile(cput[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'))
  fig_errc_a.line(np.percentile(csize[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'), line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_errc_a.line(np.percentile(csize[aidx,:,:], 50, axis=0), np.percentile(scaled_err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, line_dash='dotted')
  fig_errt_a.line(np.percentile(cput[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm.decode('utf-8'), line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_errt_a.line(np.percentile(cput[aidx,:,:], 50, axis=0), np.percentile(scaled_err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, line_dash='dotted')
 
for f in [fig_err_g, fig_err_a, fig_csz_a, fig_csz_g, fig_t_a, fig_t_g, fig_errc_g, fig_errc_a, fig_errt_g, fig_errt_a]:
  f.legend.label_text_font_size= legend_font_size
  f.legend.glyph_width=100
  f.legend.glyph_height=40
  f.legend.spacing=20

fig_err_a.legend.location = 'bottom_left'
fig_csz_a.legend.location = 'bottom_right'

bkp.show(bkl.gridplot([[fig_err_g, fig_csz_g], [fig_t_g], [fig_errc_g, fig_errt_g], [fig_err_a, fig_csz_a], [fig_t_a], [fig_errc_a, fig_errt_a]]))


