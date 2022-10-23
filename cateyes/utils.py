# -*- coding: utf-8 -*-
# (c) Dirk Gütlin, 2021. <dirk.guetlin@gmail.com>
#
# License: BSD-3-Clause

"""
cateyes.utils provides utility functions to convert and handle data.
"""

import numpy as np
import warnings

WARN_SFREQ = "\n\nIrregular sampling rate detected. This can lead to impaired " \
            "performance with this classifier. Consider resampling your data to " \
            "a fixed sampling rate. Setting sampling rate to average sample difference."


def sample_data_path(name):
    """return the static path to a CatEyes sample dataset.
    
    Parameters
    ----------
    name : str
        The example file to load. Possible names are: 'example_data',
        'example_events' and 'test_data_full'.
   
    Returns
    -------
    data_path : str
        The absolute path leading to the respective .csv file on your 
        machine.
        """
    import os.path as op
    data_dir = op.join(op.dirname(__file__), "data")
    data_path = op.join(data_dir, name + ".csv")
    return op.abspath(data_path)


def discrete_to_continuous(times, discrete_times, discrete_values):
    """Matches an array of discrete events to a continuous time series.
    
    Parameters
    ----------
    times : array of (float, int)
        A 1D-array representing the sampling times of the continuous 
        eyetracking recording.
    discrete_times : array of (float, int)
        A 1D-array representing discrete timepoints at which a specific
        event occurs. Is used to map `discrete_values` onto `times`.
    discrete_values : array
        A 1D-array containing the event description or values 
        corresponding to `discrete_times`. Must be the same length as 
        `discrete_times`.
        
    Returns
    -------
    indices : array of int
        Array of length len(times) corresponding to the event index 
        of the discrete events mapped onto the sampling times.
    values : array
        Array of length len(times) corresponding to the event values
        or descriptions of the discrete events.
        
    Example
    --------
    >>> times = np.array([0., 0.1, 0.2])
    >>> dis_times, dis_values = [0.1], ["Saccade"]
    >>> discrete_to_continuous(times, dis_times, dis_values)
    array([0., 1., 1.]), array([None, 'Saccade', 'Saccade'])
    """
    
    # sort the discrete events by time
    time_val_sorted = sorted(zip(discrete_times, discrete_values))
    
    # fill the time series with indices and values
    indices = np.zeros(len(times))
    values = np.empty(len(times), dtype=object)
    for idx, (dis_time, dis_val) in enumerate(time_val_sorted):
        selected = [times >= dis_time]
        indices[selected] = idx + 1
        values[selected] = dis_val
        
    return indices, values


def continuous_to_discrete(times, indices, values):
    """Matches an array of discrete events to a continuous time series.
    Reverse function of `discrete_to_continuous`.
    
    Parameters
    ----------
    times : array of (float, int)
        A 1D-array representing the sampling times of the continuous 
        eyetracking recording.
    indices : array of int
        Array of length len(times) corresponding to the event index 
        of the discrete events mapped onto the sampling times.
    values : array
        Array of length len(times) corresponding to the event values
        or descriptions of the discrete events.
   
    Returns
    -------
    discrete_times : array of (float, int)
        A 1D-array representing discrete timepoints at which a specific
        event occurs.
    discrete_values : array
        A 1D-array containing the event description or values 
        corresponding to `discrete_times`. Is the same length as 
        `discrete_times`.
    """
    
    # fill the discrete lists with events
    discrete_times = []
    discrete_values = []
    cur_idx = np.min(indices) - 1
    for time, idx, val in zip(times, indices, values):
        if idx > cur_idx:
            discrete_times.append(time)
            discrete_values.append(val)
        cur_idx = idx
    
    return discrete_times, discrete_values


def sfreq_to_times(gaze_array, sfreq, start_time=0):
    """Creates a times array from the sampling frequency (in Hertz).
    
    Parameters
    ----------
    gaze_array : array
        The gaze array (is required to infer the number of samples).
    sfreq : float
        The sampling frequency in Hz.
    start_time : float
        The time (in seconds) at which the first sample will start.
        Default = 0.
   
    Returns
    -------
    times : array of float
        A 1D-array representing the sampling times of the recording.
        """
    return np.arange(0, len(gaze_array) / sfreq, 1. / sfreq) + start_time


def coords_to_degree(x, viewing_dist, screen_max, screen_min=None):
    """Converts gaze data expressed in any flat spatial coordinates 
    (e.g. centimetres, inch, digital coordinate frames) to degrees.
    Assumes that the default gaze location is at the center of the
    screen.
    
    Parameters
    ----------
    x : array of float
        The gaze array to transform. Can be either a 1D or 2D array.
        If a 2-D array, the first dimension must correspond to the gaze 
        dimensions (e.g. x, y) and the second to the time dimension.
    viewing_dist : float
        The distance between the eye and the screen, measured in the 
        same unit as `screen_max` and `screen_min`.
    screen_max : float, tuple/list of float
        The maximum screen coordinates measured in the same unit as 
        `viewing_dist` and `screen_min`. If `x` is a 2D array, 
        `screen_max` must be an iterable of the same length as `x`. 
    screen_min : float, tuple/list of float
        The minimum screen coordinates measured in the same unit as 
        `viewing_dist` and `screen_min`. If `x` is a 2D array, 
        `screen_min` must be an iterable of the same length as `x`. 
        
    Returns
    -------
    x_converted : array of float
        The gaze array converted to degrees.
    """
    # set default for screen min
    x = np.array(x)
    if screen_min == None:
        screen_min = np.zeros_like(screen_max)
        
    # check arguments shapes
    msg_1 = "If x has more than 1 dimension, screen parameters" \
    " must be iterable objects with the same length as x."
    msg_2 = "Multiple screen parameter dimensions " \
    "were passed for only one gaze series x."
    lengthy = all([hasattr(screen_max, '__len__'),
                   hasattr(screen_min, '__len__')])
    if x.ndim > 1:
        if not lengthy:
            raise ValueError(msg_1)
        else:
            if not (len(x) == len(screen_max) == len(screen_min)):
                raise ValueError(msg_1) 
    else:
        if lengthy and not (1 == len(screen_max) == len(screen_min)):
            raise ValueError(msg_2)
    
    # convert the x array to degree using the arctan
    coord_range = np.array(screen_max) - np.array(screen_min)
    coord_range = coord_range.reshape(-1, 1)
    x = x - coord_range / 2.  # 0 should be at the center
    x = np.degrees(np.arctan2(x, viewing_dist))
    return x


def pixel_to_degree(x, viewing_dist, screen_size, screen_res):
    """Converts gaze data expressed as pixels to degrees. Assumes 
    that the default gaze location is at the center of the screen.
    
    Parameters
    ----------
    x : array of float
        The gaze array to transform. Can be either a 1D or 2D array.
        If a 2-D array, the first dimension must correspond to the gaze 
        dimensions (e.g. x, y) and the second to the time dimension.
    viewing_dist : float
        The distance between the eye and the screen, measured in the 
        same unit as `screen_size`.
    screen_size : float, tuple/list of float
        The screen size measured in the same unit as `screen_res`. 
        If `x` is a 2D array, `screen_size` must be an iterable of the
        same length as `x`.
    screen_res : float, tuple/list of float
        The screen resolution measured in the same unit as `screen_size`. 
        If `x` is a 2D array, `screen_res` must be an iterable of the
        same length as `x`.
        
    Returns
    -------
    x_converted : array of float
        The gaze array converted to degrees.
    """
    # check arguments shapes
    msg_1 = "If x has more than 1 dimension, screen_res" \
    " must be an iterable object with the same length as x."
    msg_2 = "Multiple screen_res dimensions " \
    "were passed for only one gaze series x."
    x = np.array(x)
    lengthy = hasattr(screen_res, '__len__')
    if x.ndim > 1:
        if (not lengthy) or (lengthy and len(x) != len(screen_res)):
            raise ValueError(msg_1)
    else:
        if lengthy and len(screen_max) != 1:
            raise ValueError(msg_2)

    # convert from pixels to spatial unit
    screen_size = np.array(screen_size).reshape(-1, 1)
    screen_res = np.array(screen_res).reshape(-1, 1)
    x = x / screen_res * screen_size
    
    # convert the spatial coordinates to degree
    return coords_to_degree(x, viewing_dist, screen_size)


def _get_time(x, time, warn_sfreq=False):
    """Process times argument to sfreq/times array"""
    # process time argument
    if hasattr(time, '__iter__'):
        # create sfreq from times array
        times = np.array(time)
        if warn_sfreq and (np.std(times[1:] - times[:-1]) > 1e-5):
            warnings.warn(WARN_SFREQ)
        sfreq = 1. / np.mean(times[1:] - times[:-1]) 
    else:
        # create times array from sfreq
        sfreq = time
        times = sfreq_to_times(x, sfreq)
    return times, sfreq
