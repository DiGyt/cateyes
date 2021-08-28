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
    return np.arange(0, len(gaze_array), 1. / sfreq) + start_time