# Artificial Neural Network Enhanced Bayesian PET Image Reconstruction

This repository contains the caffe version code for the paper Artificial Neural Network Enhanced Bayesian PET Image Reconstruction. 

## Usage

1. Data

* Training images, including 3 MAP reconstructed brain PET image of one subject at one count level and 1 corresponding phantom image;
* Testing images, which are 3 MAP reconstructions of another subject at another count level.
* Image size: 256 x 256 x207

2. Training and validating

* Generating training and validating data consisting of 3D image patches extracted from training images, and the corresponding label 
using “create_train_val_data_and_label.m”;
* Writing data to HDF5 using “create_train_hdf5.m” and "create_val_hdf5.m";
* ANN model: “nonlinear_train_val.prototxt”;
* Training parameters: “solver.prototxt”.

3. Testing

* ANN model: “deploy.prototxt”;
* Applying the trained model to process testing images with “MAP_ANN_testing.m”.

## Citation

If this package helps your research, please cite the following paper in your publications:

B.Yang, L. Ying, and J. Tang, “Artificial Neural Network
Enhanced Bayesian PET Image Reconstruction,” DOI 10.1109/TMI.2018.2803681, IEEE
Transactions on Medical Imaging, 2018.

## Note 

Pleas contact us to access the training and testing image files as they are too large to upload.

## Contact

jingtang at gmail dot com

byang2345 at oakland dot edu

Any discussions, suggestions, and questions are welcome.
