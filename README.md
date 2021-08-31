
<img src="/files/imgs/cateye_header.png" alt="CatEye logo" height="100"/>

___
### Categorization for Eyetracking in Python

- [Introduction](#introduction)
- [Installation](#installation)
- [Examples](#examples)
- [Documentation](#documentation)


## Introduction

This repository was developed for Peter König's Lab at the Institute of Cognitive Science, Osnabrück. Its aim is to provide easy access to different automated gaze classification algorithms and to generate a simplistic, flexible, and elegant way of handling Eyetracking data.

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

### Minimal use example
This minimal example applies the NSLR-HMM algorithm to a simple 2D gaze array and plots the results using the cateye plotting functions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/cateye/blob/main/example_minimal_use.ipynb)

___

### Pandas workflow example
This notebook gives a more extensive example on CatEye, including data organisation and manipulation with pandas (including e.g. resampling, interpolating, median-boxcar-filtering). The NSLR-HMM and REMoDNaV classification algorithms are applied and visualized using different internal and external plotting functions.

[![Open in Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://colab.research.google.com/github/DiGyt/cateye/blob/main/example_pandas_workflow.ipynb)


## Documentation

 #TODO

[Documentation](https://htmlpreview.github.io/?https://github.com/DiGyt/cateye/blob/main/documentation/cateye/index.html)

!pdoc --html --output-dir documentation cateye
