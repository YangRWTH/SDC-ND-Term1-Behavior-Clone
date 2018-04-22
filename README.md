# **Behavioral Cloning** 

## Report

**Behavioral Cloning Project**

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

[//]: # (Image References)

[image1]: ./final_report/network_architecture.png "Model Visualization"
[image3]: ./final_report/left_2016_12_01_13_31_12_937.jpg "Recovery Image"
[image4]: ./final_report/right_2016_12_01_13_31_12_937.jpg "Recovery Image"
[image5]: ./final_report/placeholder_small.png "Recovery Image"
[image6]: ./final_report/center_2016_12_01_13_31_12_937.jpg "Normal Image"
[image7]: ./final_report/flip_center_2016_12_01_13_31_12_937.jpg "Flipped Image"

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/432/view) individually and describe how I addressed each point in my implementation.  

---
### Files Submitted & Code Quality

#### 1. Submission includes all required files and can be used to run the simulator in autonomous mode

My project includes the following files:
* model.py containing the script to create and train the model
* drive.py for driving the car in autonomous mode
* model.h5 containing a trained convolution neural network 
* writeup_report.md summarizes the results
* Here is the [video](final_run.mp4) of the final result.

<iframe width="320" height="240" src="final_run.mp4" frameborder="0" allowfullscreen>

</iframe>

#### 2. Submission includes functional code
Using the Udacity provided simulator and my drive.py file, the car can be driven autonomously around the track by executing 
```sh
python drive.py model.h5
```

#### 3. Submission code is usable and readable

The model.py file contains the code for training and saving the convolution neural network. The file shows the pipeline I used for training and validating the model, and it contains comments to explain how the code works.

### Model Architecture and Training Strategy

#### 1. An appropriate model architecture has been employed

the model looks like as follows:

- Image normalization
- Convolution: 5x5, filter: 24, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 36, strides: 2x2, activation: RELU
- Convolution: 5x5, filter: 48, strides: 2x2, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Convolution: 3x3, filter: 64, strides: 1x1, activation: RELU
- Drop out (0.5)
- Fully connected: neurons: 100, activation: RELU
- Fully connected: neurons:  50, activation: RELU
- Fully connected: neurons:  10, activation: RELU
- Fully connected: neurons:   1 (output)

As per the NVIDIA model, the convolution layers are meant to handle feature engineering and the fully connected layer for predicting the steering angle.  However, as stated in the NVIDIA document, it is not clear where to draw such a clear distinction.  Overall, the model is very functional to clone the given steering behavior.  

The below is an model structure output from the Keras which gives more details on the shapes and the number of parameters.

| Layer (type)                   |Output Shape      |Params  |
|--------------------------------|------------------|-------:|
|lambda_1 (Lambda)               |(None, 66, 200, 3)|0       |
|convolution2d(24, 5, 5,) 		 |(None, 31, 98, 24)|1824    |
|convolution2d(36, 5, 5) 	     |(None, 14, 47, 36)|21636   |
|convolution2d(48, 5, 5) 		 |(None, 5, 22, 48) |43248   |
|convolution2d(64, 3, 3) 		 |(None, 3, 20, 64) |27712   |
|convolution2d(64, 3, 3) 		 |(None, 1, 18, 64) |36928   |
|dropout             			 |(None, 1, 18, 64) |0       |
|flatten            			 |(None, 1152)      |0       |
|dense(100)                      |(None, 100)       |115300  |
|dense(50)                		 |(None, 50)        |5050    |
|dense(10)                 		 |(None, 10)        |510     |
|dense(1)                 		 |(None, 1)         |11      |



#### 2. Attempts to reduce overfitting in the model

The model contains dropout layers in order to reduce overfitting (model.py lines 31). 

The model was trained and validated on different data sets to ensure that the model was not overfitting (code line 56-58). The model was tested by running it through the simulator and ensuring that the vehicle could stay on the track.

#### 3. Model parameter tuning

The model used an adam optimizer, so the learning rate was not tuned manually (model.py line 63).

#### 4. Appropriate training data

Training data was chosen to keep the vehicle driving on the road. I used a combination of center lane driving, recovering from the left and right sides of the road.Besides I added flapped images into training data. In order to gauge how well the model was working, I split my image and steering angle data into a training and validation set.

For details about how I created the training data, see the next section. 

### Model Architecture and Training Strategy

#### 1. Solution Design Approach

The overall strategy for deriving a model architecture was to train a network from scratch.

The design of the network is based on the NVIDIA model, which has been used by NVIDIA for the end-to-end self driving test. As such, it is well suited for the project. As the NVIDIA model is well documented, what I needed to do was to focus on how to create the training data and how to adjust to the model to achieve the best result, for example, 

- I used Lambda layer to normalized input images to avoid saturation and make gradients work better.
- I've added an additional dropout layer to avoid overfitting after the convolution layers



My first step was to use the images of center driving lane so that I could check that all the steps from prepossessing to running simulator worked correctly. Of course, the vehicle fell off the track in this case because of lacking of training data.

To improve the driving behavior, I added the images from side camera with corrected steering values. After that, I flipped all the images to compensate the influence of the driving direction.

Then I trained the model again and run the simulator to see how well the car was driving around track one. 

At the end of the process, the vehicle is able to drive autonomously around the track without leaving the road.

#### 2. Final Model Architecture

The final model architecture (model.py lines 24-37) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture [1], 

<p align="center">
  <img width="400" height="460" src= "./final_report/network_architecture.png" >
</p>

#### 3. Creation of the Training Set & Training Process

## Model Training

### Image Augumentation

For training, I used the following augmentation technique along with Python generator to generate unlimited number of images:

- choose all right, left or center images.
- For left image, steering angle is adjusted by +0.2
- For right image, steering angle is adjusted by -0.2
- flip all images left/right


I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to get back to the track center.

Image from center camera
![alt text][image6]

Image from left camera 
![alt text][image3]

Image from right camera 
![alt text][image4]

Flipped Image
![alt text][image7]



After the collection process, I had 48.215 data points.


I finally randomly shuffled the data set and put 20% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was 2. I used an adam optimizer so that manually training the learning rate wasn't necessary.



## References


1. NVIDIA model: https://devblogs.nvidia.com/parallelforall/deep-learning-self-driving-cars/