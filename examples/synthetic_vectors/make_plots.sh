#!/bin/bash

python3 main.py --data_type normal plot Ms err --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label Error

python3 main.py --data_type normal plot csize err --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label Error

python3 main.py --data_type normal plot cput err --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label Error


python3 main.py --data_type axis --data_num 100 --coreset_size_max 100 --coreset_num_sizes 10 plot Ms err --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Iterations" --plot_y_label Error

python3 main.py --data_type axis --data_num 100 --coreset_size_max 100 --coreset_num_sizes 10 plot csize err --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "Coreset Size" --plot_y_label Error

python3 main.py --data_type axis --data_num 100 --coreset_size_max 100 --coreset_num_sizes 10 plot cput err --summarize trial --groupby Ms --plot_type line --plot_legend alg --plot_x_label "CPU Time (s)" --plot_y_label Error
