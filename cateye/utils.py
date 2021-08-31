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
        
    Examples
    --------
    >>> times = np.array([0., 0.1, 0.2, 0.3])
    >>> dis_times, dis_values = [0.2], ["Saccade"]
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
    """Transforms an array of discrete events to a continuous time series."""
    
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
    """Creates a times array from the sampling frequency (in Hertz)."""
    return np.arange(0, len(gaze_array) / sfreq, 1. / sfreq) + start_time


def pixel_to_deg(x, screen_size, screen_res, viewing_dist, return_factor=False):
    """Converts pixels (or any other spatial gaze coordinates) to degrees."""
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
    if return_factor:
        return x * factor, factor
    else:
        return x * np.array([factor]).T
