# Traffic Sign Recognition


---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report


[//]: # (Image References)

[image1]: ./examples/visualization.jpg "Visualization"
[image2]: ./examples/grayscale.jpg "Grayscaling"
[image3]: ./examples/random_noise.jpg "Random Noise"
[image4]: ./examples/placeholder.png "Traffic Sign 1"
[image5]: ./examples/placeholder.png "Traffic Sign 2"
[image6]: ./examples/placeholder.png "Traffic Sign 3"
[image7]: ./examples/placeholder.png "Traffic Sign 4"
[image8]: ./examples/placeholder.png "Traffic Sign 5"

## Rubric Points
Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

You're reading it! and here is a link to my [project code](https://github.com/wizardkeven/Guokai-Traffic-Sign-Classifier-Project.git/Traffic_Sign_Classifier.ipynb)

### Data Set Summary & Exploration

As the first step, I should make clear what the given data look like as Q&A below. 
- What are the data package? 
	- It is consisted of three files: train, valid and test.
- What's inside of these files? 
	- images and labels. 
- What's the format of this files?
	- images: 32x32x3 RGB images
	- lables: 0 ~ 42 integers representing 43 types traffic signs
- How are each type of these signs distributed?
	- To make it straight-forward, I use histograph to visualize the distributions of these sighs(See in the ipython files). Obviousely, theses signs distribute unevenly e.t.c there are 7 times of size difference between the smallest sign samples and the largest sign samples. That may lead sample deviations and recognition errors to these small sign sets.

### Design and Test a Model Architecture

1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. (OPTIONAL: As described in the "Stand Out Suggestions" part of the rubric, if you generated additional data for training, describe why you decided to generate additional data, how you generated the data, and provide example images of the additional data. Then describe the characteristics of the augmented training set like number of images in the set, number of images for each class, etc.)

#### 1. Color space transform and normalization

Before starting training, I was led to image preprocessing according to the flow and the notices given by ipython file. But later I found this quite misleading. Because 
1. All the given images have been adjusted to 32x32x3 RGB images and all the labels are numbers between 0~43. As we know, preprocessing of data is only needed when we prove that the current data can't fit the training standard or they work evidently bad on starting training model. In other word, data preprocessing serves for training model and we do it only at time when we really need it. Otherwise, I think it will be a waste of time to explore the image preprocessing techniques before get hand on the model training.
2. As we haven't really start building model, we have no idea which method gives better performance.

But still, I started trying like grayscale, standard normalization as suggested like (pixels-128)/128, and visualize some images in the training set. But I got back and forth later trying to get better performance out from different data preprocessing like RGB to YUV and normalizing to images with pixels around 0.5 and 0 etc. I observed that these methods can help normalize the luminance of some images which are hard to identify due to the darkness. But these methods has little effect on improving predicting accuracy or even negative effect - lower the accuracy of both training and validation. I figured out later the possible reason of this may be that the preprocessing can only help to improve the model when we come into bottleneck of improving the accuracy of a model and this preprocessing method is matching the architecture and mechanism of your model. For example, I saw a model in which there three Neural network layers using 1x1 filters in the beginning and the author explains these three layers can help learn the color spaces of RGB images. So if we apply this model with extra image preprocessing of color space transform it will probably redundant or downgrade the model effectiveness. So I decided to apply an architecture and at least to start training on these images to see what further methods will be needed to improve the model. 

#### 2. Image augmentation

I realized that I may need this only after I failed in improve the performance of the then model. It is essential to train a model with data as big as possible. But it is also hard and expensive to get more data. The common way to avoid these headache is apply data some method to augment the disposable data.

At beginning I was thinking there may exist some handy image augmentation libraries with finely tuned parameters such as rotation angles, resizing factors etc. But I ended up not finding a single perfect library. So I had to do it my self. 

I applied random image resizing, translating, rotation, flippng and add salt-pepper noise(later also add brightness augmentation ). I tried to augment with a factor of 10 or 5(means generate 10 more or 5 more images for each given image in train.p) and used the augmented training data to feed the LeNet model along with step 1(color-space transform and pixels normalization). But still got little improvement of performance or even worse no matter how many times I trained with different parameters. 

I finally realized it may result from the augmentaton itself. As the generated images are 5 or 10 times more than the original data samples, these images may mislead the model to get wrong parameters if these images are generated with inappropriate parameters. In other words, it is better to use a relatively small but "correct" data set than a big and inappropriately augmented data set. For example, the generated images with a cropped part of a 2 times resizing and a 15 degree rotating will not appear in the validation data nor test data even in real environment. But if I feed the model with these over-distorted images, the learned model will get wrong parameters especially for LeNet with a simple architecture and a small set of neurons and layers. 

At last, I found some useful suggestion on udacity forum. A common idea of augmenting images is to slightly adjust the given data like rotate with +/-5 degree, resize with a factor of 1.2 or around etc. I enlarge the train set with a factor 10 and concatenated with the original one. It work better after applying these suggestions. 

#### 2. Finding good architecture

##### 2.1 Apply LeNet and tune the model
As a first step, I decided to apply LeNet as a starting point architecture because this was the only architecture that I really applied and understanded. 

I implemented it as the tutorial said and trained on the given model. It yielded a validation accuracy around 0.9 or below after I tuned the parameters **_Epochs_**, **_BATCH_SIZE_**, and **_Learning_rate_** to 20(or 50 with no improvement),128,0.001. At this moment, I thought of the previous things - color-transform and normalization. So I apllied both but with no improvement even worse however I tuned with different parameters and combinations. Then I realized I may need to exend or augment the given training data to feed the model. But again, it doesn't help.

I felt helpless after trying all these. And at this time my cuda and tensorflow collapsed after an inaccidentally system upgrading. It took me nearly one week to tackle this nasty issue.

After that, I tried two other architectures published on medium:

* one with three layers 32@5x5,64@5x5,128@5x5, 3 fully-connected layers, all three layers feeding forward to fully-connected layers and brightness tweaking, 10x data augmentation - works bad on my own implementation
* one with 3@1x1 at first and the rest of layers remain the same as above, 10x data augmentation - also works bad on my own implementation

I was feeling depressed and hopeless at that time. The only thing I can do is keep digging in forum and slack. I then got more tips about tuning and training model. 
* Start with a simple and small model and train and test on a small data set
* Get more training epochs if it really help(by visualizing the improvement of performance with gradually added epochs)
* Gradually enlarge the data set and add some preprosessing and image augmentation appropriately
* If there is a big gap between training accuracy and validation accuracy
* * Training accuracy is bigger and bigger than validation accuray during training -> overfitting
* * if it is opposite -> underfitting -> keep adding more training epoches to fit the model
*  Add maxpooling, dropout to avoid overfitting
*  Add more layers and neurons if above have exhausted the potential of current model
*  Add feed-forward of previous layers to fully-connected layers to give more feature information of first few layers

After adopting these tips, my model finally yielded a validation accuracy of 0.97 and 0.96 for test accuracy. It is still not the state-of-the-art performance, but I know the directions to finetune the model to get better after this process.

##### 2.2 Final model architecture model

My final model consisted of the following layers:

| Layer         		|     Description	        					| 
|:---------------------:|:---------------------------------------------:| 
| Input         		| 32x32x1 RGB image   							| 
| Convolution 5x5     	| 1x1 stride, same padding, outputs 32x32x8 	|
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 16x16x8 				    |
| Dropout				| keep_prob: 0.9								|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 16x16x16    |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 8x8x16 				    |
| Dropout				| keep_prob: 0.8								|
| Convolution 5x5	    | 1x1 stride, same padding, outputs 8x8x64      |
| RELU					|												|
| Max pooling	      	| 2x2 stride,  outputs 4x4x64 				    |
| Dropout				| keep_prob: 0.7								|
| Fully connected		| 1408:4xMP of 1st,2xMP of 2nd,3rd layer, output 172|
| Dropout				| keep_prob: 0.5								|
| Fully connected		| output 86										|
| Dropout				| keep_prob: 0.5								|
| Fully connected		| output 43 									|
| Softmax				|        									|
 


#### 3 Trained the model

With all the trials above, I get the model trained with less problems such as learning rate, epochs and batch size. I observed the losses of both train and validation for each epoch using a plotting method and tuned the keep_prob when I observed the overfitting. The numbers of neurons are found out when I simply augment the numbers of the neurons for deep layers and to make the number as a base of 8. This may not make sense, but I can get acceptable performance for this project. I can get better output if I keep tuning the model if I have time.

My final model results were:
* training set accuracy of 0.98
* validation set accuracy of 0.97
* test set accuracy of 0.96

If an iterative approach was chosen:
* What was the first architecture that was tried and why was it chosen?
* * LeNet. Because it is simple and I am fimiliar with it after the course.
* What were some problems with the initial architecture?
* * too simple architecture with little layers and neurons and also without dropout layers to avoid overfitting, which can not learn enough features from the train data
* How was the architecture adjusted and why was it adjusted? Typical adjustments could include choosing a different model architecture, adding or taking away layers (pooling, dropout, convolution, etc), using an activation function or changing the activation function. One common justification for adjusting an architecture would be due to overfitting or underfitting. A high accuracy on the training set but low accuracy on the validation set indicates over fitting; a low accuracy on both sets indicates under fitting.
* * Obviousely I adapted all these ideas to adjust the architecture.
* Which parameters were tuned? How were they adjusted and why?
* * Epoch: 50, learning rate: 0.001, batch_size: 128. The batch_size works little for performance improvement, so I fixed it with accordance to my GPU capacity; learning rate adn epochs are tuned due to the analyzing the output performance.
* What are some of the important design choices and why were they chosen? For example, why might a convolution layer work well with this problem? How might a dropout layer help with creating a successful model?
* * Start simple and improve gradually. 
* * Tune the model according to the observation and analyse of the output result and plot them clearly

If a well known architecture was chosen:
* What architecture was chosen?
* * A similar architecture of LeCun's to which udacity recommand from personal experiments
* Why did you believe it would be relevant to the traffic sign application?
* * because this project contains many catagories of signs and this model was responsive to this project.
* How does the final model's accuracy on the training, validation and test set provide evidence that the model is working well?
* * This is evident due to the output of the test data.
 

### Test a Model on New Images
The rest of the questions are all covered in the ipython files.


