# iTentformer: Observe the Distributional Differences in Short-Term Historical Behaviors

This repository contains the algorithm done in the
work [Vessel Trajectory Prediction in Intersection Waterways by Observing the Distributional Differences of Short-Term Historical Behaviors](https://github.com/dengfa02/iTentformer)
by Chuiyi Deng et al.

The core steps of iTentformer algorithm for the training and testing of trajectory prediction are listed in the
file.

## Domains and Datasets

**Update**: The code should be directly runnable with Python 3.x. The older versions of Python are no longer supported.
Scipy error may be displayed during runtime, just update it to the latest version (e.g. 1.11.2).

The dataset folder of this repository provides two original trajectories 245539000_ori and 410050325_ori as examples,
which are in different seas.

## Usage

To run GSVD algorithm on the task, one only need to run `GSVD_test.py`. You can also set the hyperparameters you want in
the main function of this .py file. 
