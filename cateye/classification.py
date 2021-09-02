# -*- coding: utf-8 -*-
# (c) Dirk Gütlin, 2021. <dirk.guetlin@gmail.com>
#
# License: BSD-3-Clause

"""
In cateye.classification you can find all available classification 
algorithms.
"""

import numpy as np
import nslr_hmm
from remodnav.clf import EyegazeClassifier
from .utils import discrete_to_continuous, continuous_to_discrete

import warnings

WARN_SFREQ = "\n\nIrregular sampling rate detected. This can lead to impaired " \
            "performance with this classifier. Consider resampling your data to " \
            "a fixed sampling rate. Setting sampling rate to average sample difference."

CLASSES = {nslr_hmm.FIXATION: 'Fixation',
           nslr_hmm.SACCADE: 'Saccade',
           nslr_hmm.SMOOTH_PURSUIT: 'Smooth Pursuit',
           nslr_hmm.PSO: 'PSO',
           None:"None",}

REMODNAV_CLASSES = {"FIXA":"Fixation", "SACC":"Saccade",
                    "ISAC":"ISaccade", "PURS":"Smooth Pursuit",
                    "HPSO":"High-Velocity PSO" ,
                    "LPSO":"Low-Velocity PSO",
                    "IHPS":"High-Velocity PSO (NCB)",
                    "ILPS":"Low-Velocity PSO (NCB)"}

REMODNAV_SIMPLE = {"FIXA":"Fixation", "SACC":"Saccade",
                   "ISAC":"Saccade", "PURS":"Smooth Pursuit",
                   "HPSO":"PSO" , "LPSO":"PSO",
                   "IHPS":"PSO", "ILPS":"PSO"}
    
    
def classify_nslr_hmm(x, y, time, return_discrete=False, return_orig_output=False, **nslr_kwargs):
    """Uses NSLR-HMM to predict gaze and returns segments and predicted classes.
    
    For reference see:
    Pekkanen, J., & Lappi, O. (2017). A new and general approach to 
    signal denoising and eye movement classification based on segmented 
    linear regression. Scientific reports, 7(1), 1-13.
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data. Must be 
        represented in degree units.
    y : array of float
        A 1D-array representing the y-axis of your gaze data. Must be 
        represented in degree units.
    times : array of float
        A 1D-array representing the sampling times of the continuous 
        eyetracking recording (in seconds).
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the `times` array). Default=False.
    return_orig_output : bool
        If True, additionally return NSLR-HMM's original segmentation 
        object as output. Default=False.
    **nslr_kwargs
        Any additional keyword argument will be passed to 
        nslr_hmm.classify_gaze().
        
    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event 
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
    segment_dict : dict
        A dictionary containing the original output from NSLR-HMM: 
        "sample_class", "segmentation" and "seg_class". Only returned if 
        `return_orig_output = True`.  
    
    """
    
    # extract gaze and time array
    gaze_array = np.vstack([x, y]).T
    time_array = np.array(time) if hasattr(time, '__iter__') else np.arange(0, len(x), 1/time)
    
    # classify using NSLR-HMM
    sample_class, seg, seg_class = nslr_hmm.classify_gaze(time_array, gaze_array,
                                                          **nslr_kwargs)
    
    # define discrete version of segments/classes
    segments = [s.t[0] for s in seg.segments]
    classes = seg_class
    
    # convert them if continuous series wanted
    if return_discrete == False:
        segments, classes = discrete_to_continuous(time_array, segments, classes)
    
    # add the prediction to our dataframe
    classes = [CLASSES[i] for i in classes]
    
    if return_orig_output:
        # create dictionary from it
        segment_dict = {"sample_class": sample_class, "segmentation": seg, "seg_class":seg_class}
        return segments, classes, segment_dict
    else:
        return segments, classes
    
    
def classify_remodnav(x, y, time, px2deg, return_discrete=False, return_orig_output=False,
                      simple_output=False, classifier_kwargs={}, preproc_kwargs={},
                      process_kwargs={}):
    """Uses REMoDNaV to predict gaze and returns segments and predicted classes.
    
    For reference see:
    Dar *, A. H., Wagner *, A. S. & Hanke, M. (2019). REMoDNaV: 
    Robust Eye Movement Detection for Natural Viewing. bioRxiv. 
    DOI: 10.1101/619254
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze 
        arrays or a float/int that represents the sampling rate.
    px2deg : float
        The ratio between one pixel in the recording and one degree. 
        If `x` and `y` are in degree units, px2deg = 1.
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.
    return_orig_output : bool
        If True, additionally return REMoDNaV's original segmentation 
        events as output. Default=False.
    simple_output : bool
        If True, return a simplified version of REMoDNaV's output, 
        containing only the gaze categories: ["Fixation", "Saccade",
        "Smooth Pursuit", "PSO"]. Default=False.
    classifier_kwargs : dict
        A dict consisting of keys that can be fed as keyword arguments 
        to remodnav.clf.EyegazeClassifier(). Default={}.
    preproc_kwargs : dict
        A dict consisting of keys that can be fed as keyword arguments 
        to remodnav.clf.EyegazeClassifier().preproc(). Default={}.
    process_kwargs : dict
        A dict consisting of keys that can be fed as keyword arguments 
        to remodnav.clf(). Default={}.
        
    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event 
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
    events : array
        A record array containing the original output from REMoDNaV.
        Only returned if `return_orig_output = True`.
    
    """
    
    # process time argument
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1]) 
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time
    
    # format and preprocess the data
    data = np.core.records.fromarrays([x, y], names=["x", "y"])
    
    # define the classifier, preprocess data and run the classification
    clf = EyegazeClassifier(px2deg, sfreq, **classifier_kwargs)
    data_preproc = clf.preproc(data, **preproc_kwargs)
    events = clf(data_preproc, **process_kwargs)
    
    # add the start time offset to the events
    for i in range(len(events)):
        events[i]['start_time'] += times[0]
        events[i]['end_time'] += times[0]
        
    # extract the classifications
    class_dict = REMODNAV_SIMPLE if simple_output else REMODNAV_CLASSES
    segments, classes = zip(*[(ev["start_time"], class_dict[ev["label"]]) for ev in events])
    
    # convert them if continuous series wanted
    if return_discrete == False:
        segments, classes = discrete_to_continuous(times, segments, classes)
    
    # return
    if return_orig_output:
        return segments, classes, events
    else:
        return segments, classes

    
def classify_velocity(x, y, time, threshold, return_discrete=False):
    """"Uses I-VT velocity algorithm from Salvucci & Goldberg (2000)
    to predict Saccades and returns segments and predicted classes.
    
    For reference see:
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
    and saccades in eye-tracking protocols. In Proceedings of the 
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze 
        arrays or a float/int that represents the sampling rate.
    threshold : float
        The maximally allowed velocity after which a sample should be 
        classified as "Saccade". Threshold can be interpreted as
        `gaze_units/ms`, with `gaze_units` being the spatial unit of 
        your eyetracking data (e.g. pixels, cm, degrees).
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.
        
    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event 
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
        """
    # process time argument and calculate sample threshold
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1]) 
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time
    sample_thresh = sfreq * threshold / 1000
    
    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([[0.], vels])
    
    # define classes by threshold
    classes = np.empty(len(x), dtype=object)
    classes[:] = "Fixation"
    classes[vels > sample_thresh] = "Saccade"
    
    # group consecutive classes to one segment
    segments = np.zeros(len(x), dtype=int)
    for idx in range(1, len(classes)):
        if classes[idx] == classes[idx - 1]:
            segments[idx] = segments[idx - 1]
        else:
            segments[idx] = segments[idx - 1] + 1
    
    # return output
    if return_discrete:
        segments, classes = continuous_to_discrete(times, segments, classes)     
    return segments, classes
    
    
def classify_dispersion(x, y, time, threshold, window_len, return_discrete=False):
    """Uses I-DT dispersion algorithm from Salvucci & Goldberg (2000)
    to predict Fixations and returns segments and predicted classes.
    
    For reference see:
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
    and saccades in eye-tracking protocols. In Proceedings of the 
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze 
        arrays or a float/int that represents the sampling rate.
    threshold : float
        The maximally allowed dispersion (difference of x/y min and 
        max values) within `window_len` in order to be counted as a 
        Fixation. Value depends on the unit of your gaze data.
    window_len : float
        The window length in seconds within which the dispersion is 
        calculated.
    return_discrete : bool
        If True, returns the output in discrete format, if False, in
        continuous format (matching the gaze array). Default=False.
        
    Returns
    -------
    segments : array of (int, float)
        Either the event indices (continuous format) or the event 
        times (discrete format), indicating the start of a new segment.
    classes : array of str
        The predicted class corresponding to each element in `segments`.
    """

    def _disp(win_x, win_y):
        """Calculate the dispersion of a window."""
        delta_x = np.max(win_x) - np.min(win_x)
        delta_y =np.max(win_y) - np.min(win_y)
        return delta_x + delta_y
    
    # process time argument
    if hasattr(time, '__iter__'):
        times = np.array(time)
        if np.std(times[1:] - times[:-1]) > 1e-5:
            warnings.warn(WARN_SFREQ)
        sfreq = 1 / np.mean(times[1:] - times[:-1]) 
    else:
        times = np.arange(0, len(x), 1 / time)
        sfreq = time
    
    # infer number of samples from windowlen
    n_samples = int(sfreq * window_len)

    # per default everything is a saccade
    segments = np.zeros(len(x), dtype=int)
    classes = np.empty(len(x), dtype=object)
    classes[0:] = "Saccade"
    
    # set start window and segment
    i_start = 0
    i_stop = n_samples
    seg_idx = 0
    
    while i_stop <= len(x):
        
        # set the current window
        win_x = x[i_start:i_stop]
        win_y = y[i_start:i_stop]
        
        # if we're in a Fixation
        if _disp(win_x, win_y) <= threshold:
            
            # start a fixation segment
            seg_idx += 1
            
            # as long as we're in the fixation
            while _disp(win_x, win_y) <= threshold and i_stop < len(x):
                
                # make the chunk larger
                i_stop += 1
                win_x = x[i_start:i_stop]
                win_y = y[i_start:i_stop]
            
            # mark it
            classes[i_start:i_stop] = "Fixation"
            segments[i_start:i_stop] = seg_idx
            
            # start looking at a new chunk
            i_start = i_stop
            i_stop = i_stop + n_samples
            seg_idx += 1
            
        else:
            # move window point further
            segments[i_start:i_stop] = seg_idx
            i_start += 1
            i_stop = i_start + n_samples
    
    # return output
    if return_discrete:
        segments, classes = continuous_to_discrete(times, segments, classes)
    return segments, classes