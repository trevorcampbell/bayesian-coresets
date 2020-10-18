#!/bin/bash

python3 main.py plot --plot_legend alg_nm --plot_x Ms --plot_y err

python3 main.py plot --plot_legend alg_nm --plot_x csize --plot_y err

python3 main.py plot --plot_legend alg_nm --plot_x cput --plot_y err
