# Trend-Enhanced Variate Transformer for Vessel Trajectory Prediction by Exploiting Short-Term Behavior Distribution Differences at Intersections

This repository contains the algorithm done in the
work [Trend-Enhanced Variate Transformer for Vessel Trajectory Prediction by Exploiting Short-Term Behavior Distribution Differences at Intersections](https://github.com/dengfa02/iTentformer)
by Chuiyi Deng et al. This paper will be published in `IEEE TIM 2025`.

The core steps of iTentformer algorithm for the training and testing of trajectory prediction are listed in the
file. If you find this paper inspiring, please cite the following format:
```
@article{deng2025trend,
  title={Trend-Enhanced Variate Transformer for Vessel Trajectory Prediction by Exploiting Short-Term Behavior Distribution Differences at Intersections},
  author={Deng, Chuiyi and Wang, Shuangxin and Li, Junwei and Liu, Jingyi and Li, Hongrui and Zhao, Zhuoyi and Guo, Yanyin and Song, Mingli},
  journal={IEEE Transactions on Instrumentation and Measurement},
  year={2025},
  publisher={IEEE}
}

@article{deng2023GSVD,
  title={Graph Signal Variation Detection: A novel approach for identifying and reconstructing ship AIS tangled trajectories},
  author={Deng, Chuiyi and Wang, Shuangxin and Liu, Jingyi and Li, Hongrui and Chu, Boce and others},
  journal={Ocean Engineering},
  volume={286},
  pages={115452},
  year={2023},
  publisher={Elsevier}
}
```
[Code of GSVD](https://github.com/dengfa02/Graph-Signal-Variation-Detection-GSVD)

## Domains and Datasets

**Update**: The code should be directly runnable with Python 3.x. The older versions of Python are no longer supported.
Scipy error may be displayed during runtime, just update it to the latest version (e.g. 1.11.2).

The dataset folder of this repository provides a 'example_bohai' dataset.

## Usage

To run algorithm on the task, one only need to run `iTentformer.py`. You can also set the hyperparameters you want in
the main function of this .py file. 

## Acknowledge

The main experimental part of this paper is completed in Beijing Jiaotong University.

