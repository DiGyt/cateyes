# -*- coding: utf-8 -*-
# (c) Dirk Gütlin, 2021. <dirk.guetlin@gmail.com>
#
# License: BSD-3-Clause

"""
In cateyes.classification you can find all available classification 
algorithms.
"""

import numpy as np
import nslr_hmm
from remodnav.clf import EyegazeClassifier
from .utils import discrete_to_continuous, continuous_to_discrete, _get_time

CLASSES = {nslr_hmm.FIXATION: 'Fixation',
           nslr_hmm.SACCADE: 'Saccade',
           nslr_hmm.SMOOTH_PURSUIT: 'Smooth Pursuit',
           nslr_hmm.PSO: 'PSO',
           None:"None",}

REMODNAV_CLASSES = {"FIXA":"Fixation", "SACC":"Saccade",
                    "ISAC":"Saccade (ISI)", "PURS":"Smooth Pursuit",
                    "HPSO":"High-Velocity PSO" ,
                    "LPSO":"Low-Velocity PSO",
                    "IHPS":"High-Velocity PSO (ISI)",
                    "ILPS":"Low-Velocity PSO (ISI)"}

REMODNAV_SIMPLE = {"FIXA":"Fixation", "SACC":"Saccade",
                   "ISAC":"Saccade", "PURS":"Smooth Pursuit",
                   "HPSO":"PSO" , "LPSO":"PSO",
                   "IHPS":"PSO", "ILPS":"PSO"}
    
    
def classify_nslr_hmm(x, y, time, return_discrete=False, return_orig_output=False, **nslr_kwargs):
    """Robust gaze classification using NSLR-HMM by Pekannen & Lappi (2017).
    
    NSLR-HMM takes eye tracking data (in degree units), segments them using 
    Naive Segmented Linear Regression and then categorizes these segments based 
    on a pretrained Hidden Markov Model. NSLR-HMM can separate between the 
    following classes:
    ```
    Fixation, Saccade, Smooth Pursuit, PSO
    ```
    
    For more information and documentation, see the [pupil-labs implementation].
    [pupil-labs implementation]: https://github.com/pupil-labs/nslr-hmm
    
    For reference see:
    
    ---
    Pekkanen, J., & Lappi, O. (2017). A new and general approach to 
    signal denoising and eye movement classification based on segmented 
    linear regression. Scientific reports, 7(1), 1-13.
    ---
    
    
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
    time_array, sfreq = _get_time(x, time)
    
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
    """REMoDNaV robust eye movement prediction by Dar, Wagner, & Hanke (2021).
    
    REMoDNaV is a fixation-based algorithm which is derived from the Nyström & Holmqvist 
    (2010) algorithm, but adds various extension. It aims to provide robust 
    classification under different eye tracking settings. REMoDNaV can separate between 
    the following classes:
    ```
    Fixation, Saccade, Saccade (intersaccadic interval), Smooth Pursuit,
    High-Velocity PSO, High-Velocity PSO (intersaccadic interval),
    Low-Velocity PSO, Low-Velocity PSO (intersaccadic interval)
    ```
    For information on the difference between normal (chunk boundary) intervals and 
    intersaccadic intervals, please refer to the original paper.
    
    
    If `simple_output=True`, REMoDNaV will separate between the following classes:
    ```
    Fixation, Saccade, Smooth Pursuit, PSO
    ```
    
    For more information and documentation, see the [original implementation].
    [original implementation]: https://github.com/psychoinformatics-de/remodnav
    
    
    For reference see:
    
    ---
    Dar, A. H., Wagner, A. S., & Hanke, M. (2021). REMoDNaV: robust eye-movement 
    classification for dynamic stimulation. Behavior research methods, 53(1), 399-414.
    DOI: 10.3758/s13428-020-01428-x
    ---
    Nyström, M., & Holmqvist, K. (2010). An adaptive algorithm for fixation, saccade,
    and glissade detection in eyetracking data. Behavior research methods, 
    42(1), 188-204.
    ---
    
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
    times, sfreq = _get_time(x, time, warn_sfreq=True)
    
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
    """I-VT velocity algorithm from Salvucci & Goldberg (2000).
    
    One of several algorithms proposed in Salvucci & Goldberg (2000),
    the I-VT algorithm classifies samples as saccades if their rate of
    change from a previous sample exceeds a certain threshold. I-VT 
    can separate between the following classes:
    ```
    Fixation, Saccade
    ```
    
    For reference see:
    
    ---
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
    and saccades in eye-tracking protocols. In Proceedings of the 
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    ---
    
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
        `gaze_units/s`, with `gaze_units` being the spatial unit of 
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
    times, sfreq = _get_time(x, time, warn_sfreq=True)
    sample_thresh = threshold / sfreq
    
    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([vels, [0.]])
    
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
    """I-DT dispersion algorithm from Salvucci & Goldberg (2000).
    
    The I-DT algorithm classifies fixations by checking if the dispersion of 
    samples within a certain window does not surpass a predefined threshold.
    I-DT can separate between the following classes:
    ```
    Fixation, Saccade
    ```
    
    For reference see:
    
    ---
    Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations 
    and saccades in eye-tracking protocols. In Proceedings of the 
    2000 symposium on Eye tracking research & applications (pp. 71-78).
    ---
    
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
    times, sfreq = _get_time(x, time, warn_sfreq=True)
    
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


def mad_velocity_thresh(x, y, time, th_0=200, return_past_threshs=False):
    """Robust Saccade threshold estimation using median absolute deviation.
    
    Can be used to estimate a robust velocity threshold to use as threshold
    parameter in the `classify_velocity` algorithm.
    
    Implementation taken from [this gist] by Ashima Keshava.
    [this gist]: https://gist.github.com/ashimakeshava/ecec1dffd63e49149619d3a8f2c0031f
    
    For reference, see the paper:
    
    ---
    Voloh, B., Watson, M. R., König, S., & Womelsdorf, T. (2019). MAD 
    saccade: statistically robust saccade threshold estimation via the 
    median absolute deviation. Journal of Eye Movement Research, 12(8).
    ---
    
    Parameters
    ----------
    x : array of float
        A 1D-array representing the x-axis of your gaze data.
    y : array of float
        A 1D-array representing the y-axis of your gaze data.
    time : float or array of float
        Either a 1D-array representing the sampling times of the gaze 
        arrays or a float/int that represents the sampling rate.
    th_0 : float
        The initial threshold used at start. Threshold can be interpreted 
        as `gaze_units/s`, with `gaze_units` being the spatial unit of 
        your eyetracking data (e.g. pixels, cm, degrees). Defaults to 200.
    return_past_thresholds : bool
        Whether to additionally return a list of all thresholds used 
        during iteration. Defaults do False.
        
    Returns
    -------
    threshold : float
        The maximally allowed velocity after which a sample should be 
        classified as "Saccade". Threshold can be interpreted as
        `gaze_units/ms`, with `gaze_units` being the spatial unit of 
        your eyetracking data (e.g. pixels, cm, degrees).
    past_thresholds : list of float
        A list of all thresholds used during iteration. Only returned
        if `return_past_thresholds` is True.
        
    Example
    --------
    >>> threshold = mad_velocity_thresh(x, y, time)
    >>> segments, classes = classify_velocity(x, y, time, threshold)
    """
    # process time argument and calculate sample threshold
    times, sfreq = _get_time(x, time, warn_sfreq=True)
    # get init thresh per sample
    th_0 = th_0 / sfreq
    
    # calculate movement velocities
    gaze = np.stack([x, y])
    vels = np.linalg.norm(gaze[:, 1:] - gaze[:, :-1], axis=0)
    vels = np.concatenate([[0.], vels])
    
    # define saccade threshold by MAD
    threshs = []
    angular_vel = vels
    while True:
        threshs.append(th_0)
        angular_vel = angular_vel[angular_vel < th_0]
        median = np.median(angular_vel)
        diff = (angular_vel - median) ** 2
        diff = np.sqrt(diff)
        med_abs_deviation = np.median(diff)
        th_1 = median + 3 * 1.48 * med_abs_deviation
        # print(th_0, th_1)
        if (th_0 - th_1) > 1:
            th_0 = th_1
        else:
            saccade_thresh = th_1
            threshs.append(saccade_thresh)
            break
    
    # revert units
    saccade_thresh = saccade_thresh * sfreq
    threshs = [i * sfreq for i in threshs]
    
    if return_past_threshs:
        return saccade_thresh, threshs
    else:
        return saccade_thresh
