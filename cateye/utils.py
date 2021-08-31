import numpy as np


def discrete_to_continuous(times, discrete_times, discrete_values):
    """Matches an array of discrete events to a continuous time series."""
    
    # sort the discrete events by time
    time_val_sorted = sorted(zip(discrete_times, discrete_values))
    
    # fill the time series with indices and values
    indices = np.zeros(len(times))
    values = np.empty(len(times), dtype=object)
    cur_idx = 0
    for dis_time, dis_val in time_val_sorted:
        selected = [times >= dis_time]
        indices[selected] = cur_idx
        values[selected] = dis_val
        cur_idx += 1
        
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
