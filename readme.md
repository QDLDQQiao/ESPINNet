# Enhanced Scanning Pattern-based Imaging Neural Network: ESPINNet

This repository contains an Enhanced Scanning Pattern-based Imaging Neural Network (ESPINNet) for grating interferometry and speckle tracking methods, which is based on a multi-resolution analysis of convolutional neural networks and can extract multi-contrast information including transmission, phase and dark-field.



## Network structure
![alt text](image.png)

## Installation
Use the environment.yml file to create the conda environment to run this code.
```
conda env create -f environment.yml
```

## Test data

The test data can be obtained here: https://1drv.ms/f/c/0bc6da5ca449d517/EqWzejqVtc1EjGEuOXu_ukEB_8Qy_7QPQGKOfjgH6aHVvw?e=3JKX7G

## Instructions

- 'estimate_single.py': inference example script.
- 'TensorRT_estimate_batch.py': script optimized using TensorRT to accelerate the inference speed.
- 'UMPA_script.py': script for comparison results using UMPA method from this repo:https://github.com/optimato/UMPA/tree/main 

## Acknowledgement
We thank the open-sourced UMPA package (https://github.com/optimato/UMPA/tree/main), which is well-written and optimized for parallel computing using C language.


Please cite this work when using this network!