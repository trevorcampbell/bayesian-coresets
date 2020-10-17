import bokeh.plotting as bkp
import numpy as np
import sys, os
import args
#make it so we can import models/etc from parent folder
sys.path.insert(1, os.path.join(sys.path[0], '../common'))
import plotting
import results

arguments = args.plot_parser.parse_args()

# extract the non-plotting-related arguments that the user specified (we will use these to match on when loading results)
dargs = vars(arguments)
matching_dict = {anm : dargs[anm] for anm in args.argnames if dargs[anm] is not None}
# load only the results that match (avoid high mem usage)
resdf = results.load_matching(matching_dict)

#call the generic plot function
plotting.plot(arguments, resdf)


