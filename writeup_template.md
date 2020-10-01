# **Traffic Sign Recognition** 

## Writeup

### You can use this file as a template for your writeup if you want to submit it as a markdown file, but feel free to use some other method and submit a pdf if you prefer.

---

**Build a Traffic Sign Recognition Project**

The goals / steps of this project are the following:
* Load the data set (see below for links to the project data set)
* Explore, summarize and visualize the data set
* Design, train and test a model architecture
* Use the model to make predictions on new images
* Analyze the softmax probabilities of the new images
* Summarize the results with a written report

## Rubric Points
### Here I will consider the [rubric points](https://review.udacity.com/#!/rubrics/481/view) individually and describe how I addressed each point in my implementation.  

---
### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one. You can submit your writeup as markdown or pdf. You can use this template as a guide for writing the report. The submission includes the project code.

You're reading it! The file "Traffic_Sign_Classifier.ipnyb" which is located in the same directory as this report, includes the project code.

#### 1. Provide a basic summary of the data set. In the code, the analysis should be done using python, numpy and/or pandas methods rather than hardcoding results manually.

I used the pandas library to read in the .csv file 'signnames.csv'. The reveived dataframe contains a list of the sign names and the matching IDs. 
Furthermore, I used the .shape method of the numpy arrays of the image data to get information about their size:

* The size of training set is : 34799, 32, 32, 3
* The size of the validation set is: 4410, 32, 32, 3
* The size of test set is : 12630, 32, 32, 3
* The shape of a traffic sign image is : 32,32
* The number of unique classes/labels in the data set is : 43

#### 2. Include an exploratory visualization of the dataset.

Here is an exploratory visualization of the data set. The first rows shows 5 random images of the training data, the second row shows shows 5 random images of the validation data and the third row shows 5 random images of the test data.

![](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/img_dataset.png)

Additionally, the following histograms show the distribution of the single sign types in the respective datasets.

![sign_distribution](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/sign_distribution.png)

### Design and Test a Model Architecture

#### 1. Describe how you preprocessed the image data. What techniques were chosen and why did you choose these techniques? Consider including images showing the output of each preprocessing technique. Pre-processing refers to techniques such as converting to grayscale, normalization, etc. 

As a first step, I decided to convert the images to grayscale because separate trainings and tests with rgb and gray scaled images revealed that I could achieve with the current network architecture a better accuracy with gray scaled images. It seems that the color information is not crucial for the net to classify images in this case and is even disturbing.

Here is an example of a traffic sign image before and after gray scaling.

![grayscaling](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/grayscaling.png)

As a second step, I normalized the gray scaled image data to prevent distorting differences in the ranges of values and to make the optimization process more efficient with scaled data. At this the mean value of the input data shall be roughly 0 and the variance should be equal.

Here is an example of a gray scaled traffic sign image before and after normalization (it does not make a difference for the visual content, however it does for the optimizer).

![Normalization](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/Normalization.png)



Here are some random final training and validation images after the mentioned preprocessing steps: 

![final_preprocessing](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/final_preprocessing.png)



#### 2. Describe what your final model architecture looks like including model type, layers, layer sizes, connectivity, etc.) Consider including a diagram and/or table describing the final model.

My final model consisted of the following layers:

| Layer         		|     Description	        					|
|:---------------------:|:---------------------------------------------:|
| Input         		| 32x32x1 gray scaled image |
| Convolution 5x5    | 1x1 stride, valid padding, outputs 28x28x6 |
| RELU					|												|
| Max pooling	      	| 2x2 stride, valid padding, outputs 14x14x16 |
| Convolution 5x5	  | 1x1 stride, valid padding, outputs 10x10x16 |
| RELU	|  |
| Max pooling	| 1x1 stride, valid padding, outputs 5x5x16 |
| Fully connected		| input 400, output 120 |
| Fully connected	| input 120, output 84 |
| Fully connected | input 84, output 43 (logits) |
| Softmax | input 43 (logits), output 43 (probabilities) |



#### 3. Describe how you trained your model. 

To train the model, I used the following setting:

- Optimizer: Adam Optimizer, efficient adaptive optimizer
- Loss function: Mean of cross entropy, has proven to be an efficient way for loss calculation
- Batch size: 64, figured out as a good number by try and error
- Number of epochs: 30, figured out as a good number by try and error
- Learning rate: 0.001,  starting value which has proven to be as a satisfying choice
- Keep probability for drop out: 0.5,  has proven by try and error to be a good value for successful regularization in the dense layers

#### 4. Describe the approach taken for finding a solution and getting the validation set accuracy to be at least 0.93. Include in the discussion the results on the training, validation and test sets and where in the code these were calculated. Your approach may have been an iterative process, in which case, outline the steps you took to get to the final solution and why you chose those steps. Perhaps your solution involved an already well known implementation or architecture. In this case, discuss why you think the architecture is suitable for the current problem.

My final model results were:
* validation set accuracy of  9.59
* test set accuracy of  9.41

If an iterative approach was chosen:

The first architecture which has been chosen was the LeNet from the lecture.
There were some problems with this architecture, so the expected number of channels of the input images had to be reduced from 3 to 1.
Furthermore, 10 predictions values have been output, however due to there 43 possible predictable classes, the number of output was changed to 43.
In order to prevent over fitting and to introduce regularization additional dropout layers have been implemented at the end of the each fully connected layer.
The number of epochs have been increased to 30, to train the network long enough and prevent it from under fitting. 
The batch size has been reduced to 64 due to it is more memory efficient and the accuracy has been improved.


### Test a Model on New Images

#### 1. Choose five German traffic signs found on the web and provide them in the report. For each image, discuss what quality or qualities might be difficult to classify.

Here are five German traffic signs that I found on the web:

![1](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/internet_test_data.png )



Due to all image needed to be resized to 32x32 the quality is not optimal. 
Apart from the 1. image and 4. image, there is something in the background which could disturb the classification process. 
Especially, in the last picture there is an other sign next to the Road work sign, which could adversely affect the prediction.
These test images are equally preprocessed as the training and validation images.

#### 2. Discuss the model's predictions on these new traffic signs and compare the results to predicting on the test set. At a minimum, discuss what the predictions were, the accuracy on these new predictions, and compare the accuracy to the accuracy on the test set

Here are the results of the prediction:

| Image			        |     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| Stop Sign      		| End of no passing by vehicles over 3.5 tons |
| No Entry | No entry 		|
| 30 km/h	| Yield											|
| Turn right ahead	| Children crossing	|
| Road work	| Beware of ice/snow |

The model was able to correctly guess 1 of the 5 traffic signs, which gives an accuracy of 20%. 
In comparison to accuracy of 94,1% achieved by the test data set, this is much worse. An explanation could be the loss of information by resizing the test images. Furthermore, in some images the background is additional disturbing and the signs are displayed from a view which has network has not learned to handle yet.

#### 3. Describe how certain the model is when predicting on each of the five new images by looking at the softmax probabilities for each prediction. Provide the top 5 softmax probabilities for each image along with the sign type of each probability. 

The code for making predictions on my final model is located in the section "Output Top 5 Softmax Probabilities For Each Image Found on the Web" of the Ipython notebook.

The output provides for each sign the top 5 probabilities displayed by bars:

![top5_overview](/home/fabian/CarND-Traffic-Sign-Classifier-Project/output/top5_overview.png)

The relevant values of top five soft max probabilities for test image 1 were the following:

| Probability         	|     Prediction	        					|
|:---------------------:|:---------------------------------------------:|
| .7         		| End of no passing by vehicles over 3.5 tons |
| .28     	| Priority Road |
| 0.008 | End of no passing |

Here, the network makes only wrong predictions and apparently it does not recognize any feature of a stop sign.

The relevant values of top five soft max probabilities for test image 2 were the following:

| Probability | Prediction |
| :---------: | :--------: |
|      1      |  No Entry  |

This is the only correct prediction of the network and it stands out that for this kind of sign the network is 100 % sure about its prediction.

The relevant values of top five soft max probabilities for test image 3 were the following: 

| Probability |      Prediction      |
| :---------: | :------------------: |
|     .49     |        Yield         |
|     .23     |    Priority Road     |
|     .1      |     No vehicles      |
|    0.07     |      No passing      |
|    0.03     | Go straight or right |

It is hard to say why the network made these predictions, however it does not seem to be very sure about its highest prediction. Maybe some more training data of this kind of sign image could improve the accuracy here.

The relevant values of top five soft max probabilities for test image 4 were the following:

| Probability |             Prediction              |
| :---------: | :---------------------------------: |
|    0.67     |          Children crossing          |
|    0.29     |            Priority Road            |
|    0.013    |             Ahead only              |
|    0.011    | End of all speed and passing limits |

Also here is hard to say why the network made these predictions, however it does not seem to be very sure about its highest prediction. Maybe some more training data would help, too.
It is also remarkable that it is the third test image at which the predicted sign "Priority Road" has the second highest probability although it is not the correct image. Maybe these kind of signs are too strong represented in the training data set.

The relevant values of top five soft max probabilities for test image 5 were the following:

| Probability |              Prediction               |
| :---------: | :-----------------------------------: |
|     .99     |          Beware if ice/snow           |
|     .01     | Right-of-way at the next intersection |

It stands out that the network is pretty sure about its wrong prediction of "Beware if ice/snow".  This traffic sign is also triangular, maybe this is the reason for the wrong prediction.