<img src="/docs/files/imgs/cateye_header.png" alt="CatEye logo" height="100"/>

___
### Categorization for Eyetracking in Python

- [Introduction](#introduction)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)


## Introduction

This repository was developed for Peter König's Neurobiopsychology Lab at the Institute of Cognitive Science, Osnabrück. Its aim is to provide easy access to different automated gaze classification algorithms and to generate a unified, simplistic, and elegant way of handling Eyetracking data.

Currently available gaze classification algorithms are:
- [NSLR-HMM](https://github.com/pupil-labs/nslr-hmm): Pekkanen, J., & Lappi, O. (2017). A new and general approach to signal denoising and eye movement classification based on segmented linear regression. Scientific reports, 7(1), 1-13.
- [REMoDNaV](https://github.com/psychoinformatics-de/remodnav): Dar *, A. H., Wagner *, A. S. & Hanke, M. (2019). REMoDNaV: Robust Eye Movement Detection for Natural Viewing. bioRxiv. DOI: 10.1101/619254

Of course we will aim to include more gaze classification algorithms in the future. Suggestions and links to implementations are always welcome.


## Installation

Currently, the way to install the package is:
```
pip install git+https://github.com/DiGyt/cateye.git
```
However, proper PyPI support might follow.


## Examples

CatEye is intended to work on a simple and intuitive level. This includes reducing all the overhead from external classification algorithms and relying on fundamental Python objects that can be used with whatever data format and workflow you are working.
```python
classify_nslr_hmm(times, gaze_x, gaze_y)
```

CatEye also provides simple but flexible plotting functions which can be used to visualize classified gaze data and can be further customized with matplotlib.pyplot.
```python
fig, axes = plt.subplots(2, figsize=(15, 6), sharex=True)
plot_segmentation(gaze_x, times, classification, events, show_event_text=False, 
                  show_legend=False, ax=axes[0])
plot_segmentation(gaze_y, times, classification, events, ax=axes[1])
axes[0].set_ylabel("Theta (in degree)")
axes[1].set_ylabel("Phi (in degree)")
axes[1].set_xlabel("Time in seconds");
```
<img src="/docs/files/plots/plot_segmentation.png" alt="CatEye segmentation plot" height="300"/>

To get started, we recommend going through our example notebooks. You can simply run them via your internet browser (on Google Colab's hosted runtime) by clicking on the "open in Colab" button.

### Minimal use example
This minimal example applies the NSLR-HMM algorithm to a simple 2D gaze array and plots the results using the cateye plotting functions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/cateye/blob/main/example_minimal_use.ipynb)

___

### Pandas workflow example
This notebook gives a more extensive example on CatEye, including data organisation and manipulation with pandas (including e.g. resampling, interpolating, median-boxcar-filtering). The NSLR-HMM and REMoDNaV classification algorithms are applied and visualized using different internal and external plotting functions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/cateye/blob/main/example_pandas_workflow.ipynb)


## Documentation

CatEye's documentation is created using [pdoc3](https://pdoc3.github.io/pdoc/) and [GitHub Pages](https://pages.github.com/). Click on the link below to view the documentation.

[Documentation](https://digyt.github.io/cateye/)

And remember that in most Python interfaces, you can print the docstring of a function via e.g. calling
```python
import cateye
cateye.discrete_to_continuous?
```

<!-- 
Note for myself: build the documentation with:
cd cateye_head_dir
pdoc --html --output-dir docs cateye --force 
-->
