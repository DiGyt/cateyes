<img src="/docs/files/imgs/cateye_header.png" alt="CatEyes logo" height="100"/>

___
### Simplified Categorization for Eye Tracking in Python

- [Introduction](#introduction)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)


## Introduction

This repository was developed for Peter König's Neurobiopsychology Lab at the Institute of Cognitive Science, Osnabrück. Its aim is to provide easy access to different automated gaze classification algorithms and to generate a unified, simplistic, and elegant way of handling eye tracking data.

Currently available gaze classification algorithms are:
- [REMoDNaV](https://digyt.github.io/cateyes/cateyes/classification.html#cateyes.classification.classify_remodnav): Dar *, A. H., Wagner *, A. S. & Hanke, M. (2019). REMoDNaV: Robust Eye Movement Detection for Natural Viewing. bioRxiv. DOI: 10.1101/619254
- [U'n'Eye](https://digyt.github.io/cateyes/cateyes/classification.html#cateyes.classification.classify_uneye): Bellet, M. E., Bellet, J., Nienborg, H., Hafed, Z. M., & Berens, P. (2019). Human-level saccade detection performance using deep neural networks. Journal of neurophysiology, 121(2), 646-661.
- [NSLR-HMM](https://digyt.github.io/cateyes/cateyes/classification.html#cateyes.classification.classify_nslr_hmm): Pekkanen, J., & Lappi, O. (2017). A new and general approach to signal denoising and eye movement classification based on segmented linear regression. Scientific reports, 7(1), 1-13.
- [I-DT dispersion-based algorithm](https://digyt.github.io/cateyes/cateyes/classification.html#cateyes.classification.classify_dispersion): Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. In Proceedings of the 2000 symposium on Eye tracking research & applications.
- [I-VT velocity-based algorithm](https://digyt.github.io/cateyes/cateyes/classification.html#cateyes.classification.classify_velocity): Salvucci, D. D., & Goldberg, J. H. (2000). Identifying fixations and saccades in eye-tracking protocols. In Proceedings of the 2000 symposium on Eye tracking research & applications.

Of course we will aim to include more gaze classification algorithms in the future. Suggestions and links to implementations are always welcome.


## Installation

Currently, the way to install the package is:
```
pip install git+https://github.com/DiGyt/cateyes.git
```
However, proper PyPI support might follow.


## Examples

CatEyes is intended to work on a simple and intuitive level. This includes reducing all the overhead from external classification algorithms and relying on fundamental Python objects that can be used with whatever data format and workflow you are working.
```python
classification = cateyes.classify_nslr_hmm(gaze_x, gaze_y, times)
```

CatEyes also provides simple but flexible plotting functions which can be used to visualize classified gaze data and can be further customized with matplotlib.pyplot.
```python
fig, axes = plt.subplots(2, figsize=(15, 6), sharex=True)
cateyes.plot_segmentation(gaze_x, times, classification, events, ax=axes[0],
                         show_event_text=False, show_legend=False)
cateyes.plot_segmentation(gaze_y, times, classification, events, ax=axes[1])
axes[0].set_ylabel("Theta (in degree)")
axes[1].set_ylabel("Phi (in degree)")
axes[1].set_xlabel("Time in seconds");
```
<img src="/docs/files/plots/plot_segmentation.png" alt="CatEyes segmentation plot" height="300"/>

To get started, we recommend going through our example notebooks. You can simply run them via your internet browser (on Google Colab's hosted runtime) by clicking on the "open in Colab" button.

___

### Minimal use example
This minimal example applies the NSLR-HMM algorithm to a simple 2D gaze array and plots the results using the CatEyes plotting functions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/cateyes/blob/main/example_minimal_use.ipynb)

___

### Pandas workflow example
This notebook gives a more extensive example on CatEyes, including data organisation and manipulation with pandas (including e.g. resampling, interpolating, median-boxcar-filtering). The NSLR-HMM and REMoDNaV classification algorithms are applied and visualized using different internal and external plotting functions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/cateyes/blob/main/example_pandas_workflow.ipynb)


## Documentation

CatEyes' documentation is created using [pdoc3](https://pdoc3.github.io/pdoc/) and [GitHub Pages](https://pages.github.com/). Click on the link below to view the documentation.

[Documentation](https://digyt.github.io/cateyes/)

<!-- 
Note for myself: build the documentation with:
cd cateye_head_dir
pdoc3 --html --output-dir docs cateyes -f -c sort_identifiers=False

Second Note: Deploy on PyPI like:
git clone https://github.com/DiGyt/cateyes.git
pip install cateyes/.
rm -rf dist
python cateyes/setup.py sdist
python cateyes/setup.py bdist_wheel
pip install twine
twine check dist/*
twine upload dist/*
-->
