import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.models as bkm
import bokeh.palettes 
import time
import os

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

pal = bokeh.palettes.colorblind['Colorblind'][8]
pal = [pal[0], pal[1], '#d62728', pal[3], pal[4], pal[5], pal[6], pal[7], pal[2]]


dnames = ['lr_synth', 'lr_ds1', 'lr_phishing', 'poiss_synth', 'poiss_biketrips', 'poiss_airportdelays']
algs = [('uniform', 'Uniform', pal[0]), ('hilbert','GIGA (noisy)', pal[1]), ('hilbert_corr', 'Fully Corrective GIGA (noisy)', pal[2]), ('riemann', 'Greedy', pal[3]), ('riemann_corr', 'Fully Corrective Greedy', pal[4]),('hilbert_good','GIGA (truth)', pal[5]), ('hilbert_corr_good', 'Fully Corrective GIGA (truth)', pal[6])]

fig = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL', x_axis_type='log', x_axis_label='Coreset Size')
fig2 = bkp.figure(y_axis_type='log', y_axis_label='Reverse KL', x_axis_type='log', x_axis_label='CPU Time (s)')

for f in [fig, fig2]:
  axis_font_size='12pt'
  legend_font_size='12pt'
  f.xaxis.axis_label_text_font_size= axis_font_size
  f.xaxis.major_label_text_font_size= axis_font_size
  f.yaxis.axis_label_text_font_size= axis_font_size
  f.yaxis.major_label_text_font_size= axis_font_size
  f.yaxis.formatter = logFmtr
  f.xaxis.formatter = logFmtr

dnmsalgs = [(dnm, alg) for dnm in dnames for alg in algs]

for idx, zppd in enumerate(dnmsalgs):
  dnm, alg = zppd
  trials = [fn for fn in os.listdir('.') if dnm+'_'+alg[0]+'_results_' in fn]
  if len(trials) == 0: continue
  Ms = np.load(trials[0])['Ms']
  kls = np.zeros((len(trials), len(Ms)))
  cputs = np.zeros((len(trials), len(Ms)))
  for tridx, fn in enumerate(trials):
    #np.savez(fldr+'_'+dnm+'_'+alg+'_results_'+str(ID)+'.npz', cputs=cputs, wts=wts, Ms=Ms, mus=mus_laplace, Sigs=Sigs_laplace, kls=kls_laplace)
    res = np.load(fn)
    cput = res['cputs']
    cputs[tridx, :] = cput
    wts = res['wts']
    mu = res['mus']
    Sig = res['Sigs']
    kl = res['kls']
    kls[tridx, :] = kl
    
  fig.line(Ms, kls.mean(axis=0), color=alg[2], legend=alg[1])
  fig.line(Ms, kls.mean(axis=0)+kls.std(axis=0), color=alg[2], legend=alg[1], line_dash='dashed')
  fig.line(Ms, kls.mean(axis=0)-kls.std(axis=0), color=alg[2], legend=alg[1], line_dash='dashed')

  fig2.circle(cputs.mean(axis=0), kls.mean(axis=0), color=alg[2], legend=alg[1])
  fig2.segment(x0=cputs.mean(axis=0)-cputs.std(axis=0), x1 = cputs.mean(axis=0)+cputs.std(axis=0), y0 = kls.mean(axis=0), y1 = kls.mean(axis=0), color=alg[2], legend=alg[1])
  fig2.segment(y0=kls.mean(axis=0)-kls.std(axis=0), y1 = kls.mean(axis=0)+kls.std(axis=0), x0 = cputs.mean(axis=0), x1 = cputs.mean(axis=0), color=alg[2], legend=alg[1])
   
#rndlbl = bkm.Label(x=1.0, x_offset=-10, y=700, y_units='screen', text='Full Dataset MCMC', angle=90, angle_units='deg', text_font_size='30pt')
#rndspan = bkm.Span(location = 1.0, dimension='height', line_width=8, line_color='black', line_dash='40 40')
#fig_cput.add_layout(rndspan)
#fig_cput.add_layout(rndlbl)

for f in [fig, fig2]:
  f.legend.label_text_font_size= legend_font_size
  f.legend.glyph_width=40
  f.legend.glyph_height=80
  f.legend.spacing=20
  f.legend.orientation='horizontal'

bkp.show(bkl.gridplot([[fig, fig2]]) )

