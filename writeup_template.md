**Behavioral Cloning Project**

This is my implementation of Udacity self driving car, project 3, behavioral project

This markdown lists how I arrived at my submission during my enrollment, and I improved since then

The goals / steps of this project are the following:
* Use the simulator to collect data of good driving behavior
* Build, a convolution neural network in Keras that predicts steering angles from images
* Train and validate the model with a training and validation set
* Test that the model successfully drives around track one without leaving the road
* Summarize the results with a written report

My implementation used a neural network to predict a correct steering angle, given a scene in the simulator

####1. Architecture of the neural network
*lambda: Grayscale and normalization layer: mean of RGB, then normalize to [-0.5, 0.5]
? Crop: Drop top 65 and bottom 15
? Convolution: filter size 6, kernel size 5, stride 1, padding valid
? Max pool: 2x2
? Convolution: filter size 6, kernel size 5, stride 1, padding valid
? Max pool: 2x2
? Convolution: filter size 6, kernel size 5, stride 1, padding valid
? Max pool: 2x2
? Flatten
? Dense: 40 relu neurons, normal random initialization with 0 mean 0.5 std dev. L2
regularization
? Dropout applied here
? Dense: 30 linear neurons, normal random initialization with 0 mean 0.5 std dev. L2
regularization
? Dense: 10 linear neurons, normal random initialization with 0 mean 0.5 std dev. L2
regularization
? Dense: 1 linear neurons, normal random initialization with 0 mean 0.5 std dev. L2
regularization

####2. Final Model Architecture

The final model architecture (model.py lines 18-24) consisted of a convolution neural network with the following layers and layer sizes ...

Here is a visualization of the architecture (note: visualizing the architecture is optional according to the project rubric)

![alt text][image1]

####3. Creation of the Training Set & Training Process

To capture good driving behavior, I first recorded two laps on track one using center lane driving. Here is an example image of center lane driving:

![alt text][image2]

I then recorded the vehicle recovering from the left side and right sides of the road back to center so that the vehicle would learn to .... These images show what a recovery looks like starting from ... :

![alt text][image3]
![alt text][image4]
![alt text][image5]

Then I repeated this process on track two in order to get more data points.

To augment the data sat, I also flipped images and angles thinking that this would ... For example, here is an image that has then been flipped:

![alt text][image6]
![alt text][image7]

Etc ....

After the collection process, I had X number of data points. I then preprocessed this data by ...


I finally randomly shuffled the data set and put Y% of the data into a validation set. 

I used this training data for training the model. The validation set helped determine if the model was over or under fitting. The ideal number of epochs was Z as evidenced by ... I used an adam optimizer so that manually training the learning rate wasn't necessary.
