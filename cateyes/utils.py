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

WARN_CONT = "\n\nThe discrete_times array passed to continuous_to_discrete " \
            "has the same length as the times array. Are you sure that your " \
            "discrete_times and discrete_values are not already continuous? " \
            "If they are, applying this function can lead to miscalculations." 

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
    # check to prevent passing continuous array
    if len(discrete_times) == len(times):
        warnings.warn(WARN_CONT)
    #TODO: Items will be dropped if 2 discrete events fall on one sample.
    # how to deal with this?
    
    # sort the discrete events by time
    time_val_sorted = sorted(zip(discrete_times, discrete_values),
                            key= lambda x:x[0])
    
    # fill the time series with indices and values
    indices = np.zeros(len(times))
    shape, dtype = [(x.shape, x.dtype) for x in [np.array(discrete_values)]][0]
    values = np.full((len(times),) + shape[1:], None).astype(dtype)
    for idx, (dis_time, dis_val) in enumerate(time_val_sorted):
        selected = times >= dis_time
        indices[selected] = idx + 1
        values[selected] = dis_val
        #for i in np.where(selected)[0]:  # this allows multidim arrays to pass
        #    values[i] = dis_val
        
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
    # check to prevent passing discrete array
    if any([len(times) != i for i in (len(indices), len(values))]):
        raise ValueError("Indices and values must have the " \
                         "same length as the times array.")
    
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


def get_segment_distance(x, y, times, segments, from_discrete=False,
                        return_start_end_pos=False):
    """Calculate the movement distance from start to end of a segment.
    This function can be used to calculate Saccade distance.
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    times : array of (float, int)
        A 1D-array representing the sampling times of the continuous 
        eyetracking recording.
    segments : array of (int, float)
        Either the event indices (continuous format) or the event 
        times (discrete format), indicating the start of a new segment.
    from_discrete : bool
        If True, assumes that `segments` is a discrete array and will
        also return a discrete array. Else, will treat `segments` as
        continuous array and return a continuous array. Default=False.
    return_start_end_pos : bool
        If True, additionally return the initial and final position of
        the gaze array during each segment. Default=False.
        
        
    Returns
    -------
    distances : array of float
        Array of length len(times) corresponding to the distance
        between the first and the last sample in a segment.
    start_pos : array of float
        A 2D array of shape [n_events, 2] (discrete) or 
        [n_samples, 2] (continuous) containing the initial gaze 
        positions for each segment. The second dimension of the 
        array corresponds to the x and y axis (in that order).
        Only returned if `return_start_end_pos=True`.
    end_pos : array of float
        A 2D array of shape [n_events, 2] (discrete) or 
        [n_samples, 2] (continuous) containing the final gaze 
        positions for each segment. The second dimension of the 
        array corresponds to the x and y axis (in that order).
        Only returned if `return_start_end_pos=True`.
    """
    # check if continuous segments match length
    msg = "For continuous segments, len(segments) must be equal to " \
    "len(times). If you are using a discrete segment array, please " \
    "pass `discrete=True` as an argument"
    if not from_discrete:
        if len(segments) != len(times):
            raise ValueError(msg)

        # make discrete if continuous
        segments, _ = continuous_to_discrete(times, segments, x)

    # sort the discrete events by time
    seg_times = sorted(segments)
    
    # loop over segments to find start and end positions
    start_pos, end_pos = np.zeros([2, len(seg_times), 2])  # 2 2D arrays
    nxt_seg_times = np.concatenate([seg_times[1:], [np.inf]])
    for idx, (cur, nxt) in enumerate(zip(seg_times, nxt_seg_times)):

        # select the segment in question
        selected = np.logical_and(times >= cur, times <= nxt)
        sel_x, sel_y = x[selected], y[selected]

        # add start positions and end positions to array
        start_pos[idx] = sel_x[0], sel_y[0]
        end_pos[idx] = sel_x[-1], sel_y[-1]

    #calculate distances
    distances = np.linalg.norm(end_pos - start_pos, axis=1)

    # return_start_end_pos
    out = (distances,)
    if return_start_end_pos:
        out += (start_pos, end_pos)

    # make continuous if necessary
    if not from_discrete:
        out = tuple(discrete_to_continuous(times, seg_times, i)[1] for i in out)
    return out if return_start_end_pos else out[0]


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
    coord_range = coord_range[..., np.newaxis]
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
        same unit as `screen_size` (e.g. inch or cm).
    screen_size : float, tuple/list of float
        The screen size measured in the same unit as `screen_res` (e.g. 
        inch or cm). If `x` is a 2D array, `screen_size` must be a
        list/array of the same length as `x`
    screen_res : float, tuple/list of float
        The screen resolution measured as a total of `screen_size` 
        (e.g. total number of pixels over one axis). If `x` is a 
        2D array, `screen_res` must be a list/array of the same 
        length as `x`.
        
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
        if lengthy and len(screen_res) != 1:
            raise ValueError(msg_2)

    # convert from pixels to spatial unit
    x /= np.array(screen_res)[..., np.newaxis]
    x *= np.array(screen_size)[..., np.newaxis]
    
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
