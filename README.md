# CS149b: Project 1 Room number detection and recognition 


This project implements CNN detector and recognizer. We used keras framework and opencv library to build the detector.
This detector determine digit with trained CNN classifier for the region proposed by the MSER algorithm.


## Anaconda Env

We recommend that you create and use an isolated anaconda env. You can create anaconda env for this project by following these simple steps.

* Create anaconda env with the following command line:
  * ```$ conda env create -f digit_detector.yml ```
* Activate the env
  * ```$ source activate digit_detector ```
* Run the project in this env


### Usage

The procedure to build digit detector is as follows:

#### 0. Download Dataset

Download train.tar.gz and extra.tar.gz in http://ufldl.stanford.edu/housenumbers/ and unzip the file.

There are further dataset in training the three filters which includes clips that are recorded, processed and annotated by our group that can be provided per request (in total approximately : 20.00 GB)

 

#### 1. load training samples (1_sample_loader.py)

Svhn provides cropped training samples in matlab format. 
However, it is not suitable for detecting bounding box because it introduces some distracting digits to the sides of the digit of interest. So I collected the training samples directly using full numbers images and its annotation file.


#### 2. train classifier (2_train.py)

##### 2.1. classifier used for detection

We designed a Convolutional Neural Network architecture for detecting character. This network classify text and non-text.

The architecture is as follows:

* INPUT: [32x32x1]
* CONV3-32: [32x32x32]
* CONV3-32: [32x32x32]
* POOL2: [16x16x32]
* CONV3-64: [16x16x64]
* CONV3-64: [16x16x64]
* POOL2: [8x8x64]
* FC: [1x1x1024] 
  * I used drop out in this layer.
* FC: [1x1x2]

We train in total three filters on this architecture to distinguish between text and non-text, texts and digits and specified cropped image of texts and digits. The pre-trained models are enclosed in the archive respectively as detector_model+text-nontext.hdf5, detector_model+digit-text0.hdf5, detector_model+digit-text.hdf5.
 
The accuracy of the classifier is as follows

* Training Accuracy : 98.31%
* Test Accuracy : 98.98%

* Training Accuracy : 98.51%
* Test Accuracy : 98.71%

* Training Accuracy : 41.52%
* Test Accuracy : 94.69%

##### 2.2. classifier used for recognition

This Convolutional Neural Network recognize numbers. The architecture is same except for the number of class. The pre-trained weights of the recognizor is enclosed in the archive as recognize_model.hdf5.

The architecture is as follows:

* INPUT: [32x32x1]
* CONV3-32: [32x32x32]
* CONV3-32: [32x32x32]
* POOL2: [16x16x32]
* CONV3-64: [16x16x64]
* CONV3-64: [16x16x64]
* POOL2: [8x8x64]
* FC: [1x1x1024] 
  * We used drop out in this layer.
* FC: [1x1x10]
  * number of class is 10.

The accuracy of the classifier is as follows

* Training Accuracy : 99.01%
* Test Accuracy : 99.52%


#### 3. Run the detector train models on the videos (python whole_process3.py --video <path>)

In the running time, the detector operates in the 4-steps.

1) The detector distinguish Text/Non-text through preliminary filter(detector_model+text-nontext.hdf5).

2) The detector distinguish digits from text through secondary filter(detector_model+digit-text0.hdf5). 

3) The detector filter through a specific region proposed by the MSER algorithm(detector_model+digit-text.hdf5).

4) The classifier determines whether or not it is a number in the proposed region(recognize_model.hdf5).


*The generated bounding boxed video is under the root folder named detection.avi
*The result file is under the root folder named bbox.txt


## Referred Projects

* [Yolo-digit-detector](https://github.com/penny4860/Yolo-digit-detector)

