#!/bin/bash

python3 main.py plot Ms err --summarize trial --groupby Ms --plot_toolbar --plot_type line --plot_legend alg_nm --plot_x_label "Iterations" --plot_y_label Error

python3 main.py plot csize err --summarize trial --groupby Ms --plot_type line --plot_legend alg_nm --plot_x_label "Coreset Size" --plot_y_label Error

python3 main.py plot cput err --summarize trial --groupby Ms --plot_type line --plot_legend alg_nm --plot_x_label "CPU Time" --plot_y_label Error
