"""
cateye.utils provides utility functions to convert and handle data.
"""

import numpy as np


def sample_data_path(name):
    """Load sample data. Possible names are: 'example_data', 'example_events' and 'test_data_full'."""
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
    
    # sort indices by indices
    indices, values = zip(*[i for i in sorted(zip(indices, values))])
    
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


def pixel_to_deg(x, screen_size, screen_res, viewing_dist, return_factor=False):
    """Converts pixels (or any other spatial gaze coordinates) to degrees.
    
    Parameters
    ----------
    x : array of float
        The gaze array to transform. Can be either a 1D or 2D array.
        If a 2-D array, the first dimension must correspond to the gaze 
        orientation (e.g. x, y).
    screen_size : float, tuple/list of float
        The screen size measured in the same unit as `screen_res`. 
        If x is a 2D array, `screen_size` must be an iterable of the
        same length as x.
    screen_res : float, tuple/list of float
        The screen resolution measured in the same unit as `screen_size`. 
        If x is a 2D array, `screen_res` must be an iterable of the
        same length as x.
    viewing_dist : float
        The distance between the eye and the screen, measured in the 
        same unit as `screen_size`.
    return_factor : bool
        If True, return the conversion factors additionally to the 
        converted gaze data. Default = False.
        
    Returns
    -------
    x_converted : array of float
        The gaze array converted to degrees.
    factor : array of float
        The conversion factor(s) used to convert x (with `x_converted 
        = x * factor`). Only returned if `return_factor=True`.
        """
    msg = "If x has more than 1 dimension, screen_size/screen_res " \
    "must be iterable objects with the same length as x."
    x = np.array(x)
    lengthy = all([hasattr(screen_size, '__len__'),
                   hasattr(screen_res, '__len__')])
    if x.ndim > 1:
        if lengthy:
            if not (len(x) == len(screen_size) == len(screen_res)):
                raise ValueError(msg)
        else:
            raise ValueError(msg)
    else:
        if lengthy and not (1 == len(screen_size) == len(screen_res)):
            raise ValueError("Multiple screen_size or screen_res " \
                            "were passed for only one gaze series x.")

    arctan = np.arctan2(np.array(screen_size) / 2., viewing_dist)
    factor = np.degrees(arctan / (np.array(screen_res) / 2.))
    factor = np.array([factor]).T
    if return_factor:
        return x * factor, factor
    else:
        return x * factor
