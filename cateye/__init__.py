"""
Wecome to the CatEye documentation.

In `cateye.cateye` you will find the set of available classification 
algorithms.

In `cateye.utils` you will find utility functions to convert and handle 
data.

In `cateye.visualization` you will find functions to plot and visualize
the classified Eyetracking data.

"""

from .cateye import classify_nslr_hmm, classify_remodnav
from .utils import (discrete_to_continuous, continuous_to_discrete,
                    sfreq_to_times, pixel_to_deg, sample_data_path)
from .visualization import (plot_segmentation, plot_trajectory,
                            plot_nslr_segmentation)