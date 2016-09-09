# Evaluating User Authorization Models

These files contain scripts to extract features from motion data and evaluate
models for a user a user authorization system.  The data are from 15 human
subjects performing a variety of tasks taken from the University of California
Irvine Machine Learning repository.  Time and frequency domain features are
extracted and used in various classifiers to predict an authorized user or 
unauthorized user (i.e. any other user).  An average F1 score over all users 
is computed to evaluate model performance.

## Getting Started

The index labels in the data appear to have alignment issues. Run
```
python fix_ind.py
```
to create new files with adjusted indidies.  The load_file function will use
the indicies in these files if the files exist.

To run the analysis run: 
```
python analysis.py
```
Most functions, however, will have increased functionability if run from an
interpreter.

## Prerequisites

Python 2.7
scikit-learn 0.17.1

Data files can be found at:
http://archive.ics.uci.edu/ml/datasets/Activity+Recognition+from+Single+Chest-Mounted+Accelerometer

## Acknowledgments

This project contains the peakdet function from:
https://gist.github.com/sixtenbe/1178136#file-peakdetect-py
