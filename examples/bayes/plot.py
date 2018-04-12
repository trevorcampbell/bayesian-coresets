import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.models as bkm
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

dnames_all  = [['synth', 'ds1', 'phishing'], ['synth', 'biketrips', 'airportdelays']]
fldrs = ['lr', 'poiss']

dnames_all  = [['synth']]
fldrs = ['poiss']


figs = []
for dnames, fldr in zip(dnames_all, fldrs):
  fig_F = bkp.figure(y_axis_type='log', x_axis_type='log', y_axis_label='Relative Fisher Information Distance', x_axis_label='Coreset Size', plot_width=1250, plot_height=1250)
  
  axis_font_size='36pt'
  legend_font_size='36pt'
  fig_F.xaxis.axis_label_text_font_size= axis_font_size
  fig_F.xaxis.major_label_text_font_size= axis_font_size
  fig_F.xaxis.formatter = logFmtr
  fig_F.yaxis.axis_label_text_font_size= axis_font_size
  fig_F.yaxis.major_label_text_font_size= axis_font_size
  fig_F.yaxis.formatter = logFmtr
  fig_F.toolbar.logo = None
  fig_F.toolbar_location = None
  
  pal = bokeh.palettes.colorblind['Colorblind'][8]
  pal = [pal[0], pal[1], '#d62728', pal[3], pal[4], pal[5], pal[6], pal[7], pal[2]]
  for didx, dnm in enumerate(dnames):
    
    res = np.load(fldr +'/' + dnm  + '_results.npz')
  
    Ms = res['Ms']
    Fs = res['Fs']
    cputs = res['cputs']
    csizes = res['csizes']
    Fs_full = res['Fs_full']
    cputs_full = res['cputs_full']
    anms = res['anms']
  
    for aidx, anm in enumerate(anms):
      anm = anm.decode('utf-8')
      if anm == 'FW':
        ld = 'dashed'
      elif anm == 'RND':
        ld = 'dotted'
      else:
        ld = 'solid'
      
      if anm == 'FW' or anm == 'GIGA':
        fig_F.line(np.percentile(csizes[aidx,:,:], 50, axis=0), np.percentile(Fs[aidx, :, :], 50, axis=0)/np.percentile(Fs[2, :, :], 50, axis=0), line_color=pal[didx], line_width=8, line_dash=ld, legend=dnm if ld == 'solid' else None)
          
  rndlbl = bkm.Label(x=30, y=1.0, y_offset=-50, text='Uniform Subsampling', text_font_size='30pt')
  rndspan = bkm.Span(location = 1.0, dimension='width', line_width=8, line_color='black', line_dash='40 40')
  fig_F.add_layout(rndspan)
  fig_F.add_layout(rndlbl)
  
  fig_F.legend.label_text_font_size= legend_font_size
  fig_F.legend.glyph_width=40
  fig_F.legend.glyph_height=80
  fig_F.legend.spacing=20
  fig_F.legend.orientation='horizontal'

  figs.append(fig_F)

bkp.show(bkl.row(figs))


