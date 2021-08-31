"""
In cateye.cateye you can find all available classification 
algorithms.
"""

import numpy as np
import nslr_hmm
from remodnav.clf import EyegazeClassifier
from .utils import discrete_to_continuous

import warnings

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
    """Uses NSLR-HMM to predict gaze and returns segments and predicted classes."""
    
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
                      classifier_kwargs={}, preproc_kwargs={}, process_kwargs={},
                      simple_output=False):
    """Uses Remodnav to predict gaze and returns segments and predicted classes."""
    
    times = np.array(time) if hasattr(time, '__iter__') else np.arange(0, len(x), 1/time)
    if np.std(times[1:] - times[:-1]) > 1e-5:
        warnings.warn("\n\nIrregular sampling rate detected. This can lead to impaired performance "
                      "with this classifier. Consider resampling your data to a fixed sampling "
                      "rate. Setting sampling rate to average sample difference.")
    sampling_rate = 1 / np.mean(times[1:] - times[:-1])
    
    # format and preprocess the data
    data = np.core.records.fromarrays([x, y], names=["x", "y"])
    
    # define the classifier, preprocess data and run the classification
    clf = EyegazeClassifier(px2deg, sampling_rate, **classifier_kwargs)
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