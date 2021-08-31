# this is commented text added at the init file
#
#
# some more lines just to be sure
#
"""
This is docstring text added at the init file



let's hope there's something interesting in here-

"""
from .cateye import (classify_nslr_hmm, classify_remodnav,
                     sample_data_path)
from .utils import (discrete_to_continuous, continuous_to_discrete,
                    sfreq_to_times, pixel_to_deg)
from .visualization import (plot_segmentation, plot_trajectory,
                            plot_nslr_segmentation)
