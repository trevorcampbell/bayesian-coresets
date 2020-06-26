import numpy as np
import bokeh.plotting as bkp
import bokeh.io as bki
import bokeh.layouts as bkl
from bokeh.models import FuncTickFormatter
import bokeh.palettes 
import time
import os, sys
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
from plotting import *


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
  fig_err_g.line(Ms, np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm)
  fig_csz_g.line(Ms, np.percentile(csize[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm)
  fig_t_g.line(Ms, np.percentile(cput[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm)
  fig_errc_g.line(np.percentile(csize[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm)
  fig_errt_g.line(np.percentile(cput[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm)


aa = np.load('axis_results.npz')
anms = aa['anms']
Ms = aa['Ms']
err = aa['err']
cput = aa['cput']
csize = aa['csize']
for aidx, anm in enumerate(anms):
  fig_err_a.line(Ms, np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm, line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_csz_a.line(Ms, np.percentile(csize[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm, line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_t_a.line(Ms, np.percentile(cput[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm)
  fig_errc_a.line(np.percentile(csize[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm, line_dash=[20, 30], line_dash_offset=np.random.randint(50))
  fig_errt_a.line(np.percentile(cput[aidx,:,:], 50, axis=0), np.percentile(err[aidx,:,:], 50, axis=0), line_color=pal[aidx], line_width=8, legend=anm, line_dash=[20, 30], line_dash_offset=np.random.randint(50))
 
for f in [fig_err_g, fig_err_a, fig_csz_a, fig_csz_g, fig_t_a, fig_t_g, fig_errc_g, fig_errc_a, fig_errt_g, fig_errt_a]:
  f.legend.label_text_font_size= legend_font_size
  f.legend.glyph_width=100
  f.legend.glyph_height=40
  f.legend.spacing=20

fig_err_a.legend.location = 'bottom_left'
fig_csz_a.legend.location = 'bottom_right'

bkp.show(bkl.gridplot([[fig_err_g, fig_csz_g], [fig_t_g], [fig_errc_g, fig_errt_g], [fig_err_a, fig_csz_a], [fig_t_a], [fig_errc_a, fig_errt_a]]))
#bkp.output_file('results.html')
#bkp.save(bkl.gridplot([[fig_err_g, fig_csz_g], [fig_t_g], [fig_errc_g, fig_errt_g], [fig_err_a, fig_csz_a], [fig_t_a], [fig_errc_a, fig_errt_a]]))


