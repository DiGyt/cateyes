# -*- coding: utf-8 -*-
# (c) Dirk Gütlin, 2021. <dirk.guetlin@gmail.com>
#
# License: BSD-3-Clause

"""
.. image:: ./../files/imgs/cateye_header.png

Welcome to the CatEye documentation.

You find the documentation to all available functions by clicking on
the respective submodules.

"""

from .classification import (classify_nslr_hmm, classify_remodnav, 
                             classify_dispersion, classify_velocity)
from .utils import (discrete_to_continuous, continuous_to_discrete,
                    sfreq_to_times, pixel_to_deg, sample_data_path)
from .visualization import (plot_segmentation, plot_trajectory,
                            plot_nslr_segmentation)