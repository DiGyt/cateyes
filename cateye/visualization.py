import numpy as np
import nslr_hmm
import matplotlib.pyplot as plt

COLORS = {nslr_hmm.FIXATION: 'blue',
          nslr_hmm.SACCADE: 'black',
          nslr_hmm.SMOOTH_PURSUIT: 'green',
          nslr_hmm.PSO: 'yellow', }

N_COLS = {'Fixation': 'blue',
          'Saccade': 'black',
          'ISaccade': 'gray',
          'Smooth Pursuit': 'green',
          'PSO': 'yellow',
          'High-Velocity PSO': 'gold',
          'Low-Velocity PSO': 'greenyellow',
          'High-Velocity PSO (NCB)': 'goldenrod',
          'Low-Velocity PSO (NCB)': 'yellowgreen',
          'None':'grey',}


def plot_segmentation(gaze, times, segments=None, events=None, show_event_text=True,
                      show_legend=True, color_dict=None, ax=None):
    
    if ax == None:
        ax = plt.gca()
        
    if color_dict == None:
        color_dict = N_COLS
    
    if segments == None:
        ax.plot(times, gaze)
        
        # define the values for our legend
        color_dict = {"Gaze":"blue"}
    else:
        # add multiple plot lines for each segment
        zip_seg = zip(segments[0][:-1], segments[0][1:], segments[1][:-1])
        for start, stop, cl in zip_seg:
            select = np.logical_and(times >= start, times <= stop)
            ax.plot(times[select], gaze[select], "-", c=color_dict[cl])
        
        # define the values for our legend
        color_dict = {key:val for key, val in color_dict.items() if key in segments[1]}
        
    # define the leg artists
    leg_artists = [plt.Line2D((0,1),(0,0), color=color) for color in color_dict.values()]
    leg_indicators = list(color_dict.keys())
            
    if events != None:
        # add vertical bars at timepoints listed in events
        y_pos = ax.get_ylim()[0]
        x_pos = np.diff(ax.get_xlim()) / 200  # np.mean(times[1:] - times[:-1]) * 30
        for time, val in zip(events[0], events[1]):
            ax.axvline(x=float(time), linestyle="--")
            if show_event_text:
                ax.text(float(time) + x_pos, y_pos, f" {val}", rotation=90,
                        verticalalignment='bottom', color='#1f77b4')
        
        # add legend artists for the events
        leg_artists = leg_artists + [plt.Line2D((0,1),(0,0),  linestyle='--', color='#1f77b4')]
        leg_indicators = leg_indicators + ["Events"]
        
        
    if show_legend:
        ax.legend(leg_artists, leg_indicators, loc="lower right", title="Gaze Classification")
                
    return ax


def plot_trajectory(x, y, times, segments=None, show_legend=True, show_clean=True, show_arrows=True,
                    show_dots=False, alpha_decay=0., color_dict=None, ax=None, plot_kwargs={},
                    dot_kwargs={}, arrow_kwargs={}):
    
    if show_arrows == True and show_clean == False:
        raise ValueError("If show_arrows = True, then show_clean must be True.")
    
    if "marker" not in dot_kwargs.keys():
        dot_kwargs["marker"] = "."
    if "head_width" not in arrow_kwargs.keys():
        arrow_kwargs["head_width"] = 0.1
    
    if ax == None:
        ax = plt.gca()
        
    if color_dict == None:
        color_dict = N_COLS
    
    if segments == None:
        ax.plot(x, y, **plot_kwargs)
        
        # define the values for our legend
        color_dict = {"Gaze":"blue"}
    else:
        # add multiple plot lines for each segment
        zip_seg = zip(segments[0][:-1], segments[0][1:], segments[1][:-1])
        alpha = 1
        for start, stop, cl in reversed(list(zip_seg)):
            select = np.logical_and(times >= start, times <= stop)
            x_sel = x[select]
            y_sel = y[select]
            if show_clean and cl in ['Fixation', 'Saccade', 'ISaccade'] and len(x_sel) > 0:
                if show_arrows and cl in ['Saccade', 'ISaccade']:
                    ax.arrow(x_sel[0], y_sel[0], x_sel[-1] - x_sel[0], y_sel[-1] - y_sel[0],
                             color=color_dict[cl], length_includes_head=True, alpha=alpha,
                             **arrow_kwargs)
                else:
                    ax.plot([x_sel[0], x_sel[-1]], [y_sel[0], y_sel[-1]],
                            c=color_dict[cl], alpha=alpha, **plot_kwargs)
            else:
                ax.plot(x_sel, y_sel, c=color_dict[cl], alpha=alpha, **plot_kwargs)
                
            if show_dots and len(x_sel) > 0:
                ax.plot(x_sel[-1], y_sel[-1], c=color_dict[cl], alpha=alpha, **dot_kwargs)
            if len(x_sel) > 0:
                alpha -= alpha_decay
        
        # define the values for our legend
        color_dict = {key:val for key, val in color_dict.items() if key in segments[1]}
       
    # define legend artists
    leg_artists = [plt.Line2D((0,1),(0,0), color=color) for color in color_dict.values()]
    leg_indicators = list(color_dict.keys())
          
    if show_legend:
        ax.legend(leg_artists, leg_indicators, title="Gaze Classification")
                
    return ax


def plot_nslr_segmentation(time_array, gaze_array, segmentation, seg_class, trial_info=None,
                           title=None, stimulus=None, figsize=None, blinks=None):
    """Plots a segmentation from nslr predictions."""
    VAR_NAMES = ["Theta", "Phi"]
    
    f, axes = plt.subplots(2, 1, sharex=True, figsize=figsize)
    f.subplots_adjust(hspace=0)
    f.suptitle(title)  # , fontsize=16)

    # the limits are defined as 2 std devs from mean
    y_lims = np.array(gaze_array).std(axis=0) * 2

    for idx, ax in enumerate(axes):

        # construct a plotting frame
        ax.set_ylabel(VAR_NAMES[idx] + ": Rot. in degrees")
        ax.set_ylim([-y_lims[idx], y_lims[idx]])
        ax.axhline(y=0, color="lightgray", linestyle="--")
        ax.axhline(y=-20, color="lightgray", linestyle="--")
        ax.axhline(y=20, color="lightgray", linestyle="--")


        ax.plot(time_array, gaze_array[:,idx], '.')
        if not isinstance(stimulus, type(None)):
            st_id = ["X_Position", "Y_Position"]
            ax.plot(stimulus["Timestamp"].to_numpy(), stimulus[st_id[idx]].to_numpy(), "--", color="lightblue")

        for i, seg in enumerate(segmentation.segments):
            cls = seg_class[i]
            if (seg.t[0] > time_array[0]) and (seg.t[1] < time_array[-1]): 
                if not isinstance(blinks, type(None)):
                    time_window = np.logical_and(time_array > seg.t[0], time_array < seg.t[1])
                    blink = np.mean(blinks[time_window])
                    if blink > 0.2:
                        ax.plot(seg.t, np.array(seg.x)[:, idx], color="grey")
                    else:
                        ax.plot(seg.t, np.array(seg.x)[:, idx], color=COLORS[cls])
                else:
                    ax.plot(seg.t, np.array(seg.x)[:, idx], color=COLORS[cls])

    if trial_info != None and len(trial_info) > 0:
        y_pos = - y_lims[1] * 0.95  # min([i[0] for i in gaze_array])
        x_pos = (time_array[-1] - time_array[1]) / 150

        for key in trial_info:
            axes[0].axvline(x=float(key), linestyle="--")
            axes[1].axvline(x=float(key), linestyle="--")
            axes[1].text(float(key) + x_pos, y_pos,
                         "Event: {}".format(trial_info[key]), rotation=90,
                         verticalalignment='bottom', color='#1f77b4')  #, weight="bold")

    
    leg_artists = [plt.Line2D((0,1),(0,0), marker='.', linestyle='')] + \
                  [plt.Line2D((0,1),(0,0), color=color) for color in COLORS.values()]
    leg_indicators = ["Orig. Samples", "Fixation", "Saccade", "Smooth Pursuit", "PSO"]
    plt.legend(leg_artists, leg_indicators, loc="lower right", title="Gaze Classification")
    
    if not isinstance(blinks, type(None)):
        leg_artists = leg_artists + [plt.Line2D((0,1),(0,0), color="grey")]
        leg_indicators = leg_indicators + ["Blink/Noise"]
        
    if not isinstance(stimulus, type(None)):
        leg_artists = leg_artists + [plt.Line2D((0,1),(0,0),  linestyle='--', color="lightblue")]
        leg_indicators = leg_indicators + ["Stimulus data"]
        
    plt.legend(leg_artists, leg_indicators, loc="lower right", title="Gaze Classification")
    plt.xlabel("Time in seconds")
