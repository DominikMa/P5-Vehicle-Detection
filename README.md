# **Vehicle Detection Project**

![alt text][gif_combi]


This repository contains my solution for the project "Vehicle Detection Project" of the Udacity Self-Driving Car Engineer Nanodegree Program. The main python code to run the project could be found in [P5](P5.py), the generated images in the folder [output_images](output_images/) and the videos in the folder [output_videos](output_videos/).

The following part of the README contains a writeup which describes how the vehicle detection is achieved.

---

## Writeup

[//]: # (Image References)
[gif_combi]: ./output_videos/combi_project_video.gif
[test_image1]: ./output_images/test1.jpg
[test_image2]: ./output_images/test2.jpg
[test_image3]: ./output_images/test3.jpg
[test_image5]: ./output_images/test5.jpg
[car]: ./writeup_images/car.png
[non_car]: ./writeup_images/non_car.png
[car1]: ./writeup_images/car1.png
[non_car1]: ./writeup_images/non_car1.png
[car2]: ./writeup_images/car2.png
[non_car2]: ./writeup_images/non_car2.png
[car3]: ./writeup_images/car3.png
[non_car3]: ./writeup_images/non_car3.png
[car4]: ./writeup_images/car4.png
[non_car4]: ./writeup_images/non_car4.png
[car1_hog]: ./writeup_images/car1_hog.png
[non_car1_hog]: ./writeup_images/non_car1_hog.png
[car2_hog]: ./writeup_images/car2_hog.png
[non_car2_hog]: ./writeup_images/non_car2_hog.png
[car3_hog]: ./writeup_images/car3_hog.png
[non_car3_hog]: ./writeup_images/non_car3_hog.png
[car4_hog]: ./writeup_images/car4_hog.png
[non_car4_hog]: ./writeup_images/non_car4_hog.png

### Goals

The goal of this project is to detect vehicles in images or videos and mark them with a bounding box.
The individual steps are the following:

* Preform a feature extraction from labeled images. This could include Histogram of Oriented Gradients (HOG), color and histograms of color features.
* Prepare the extracted features. For example normalize and split into train and test sets.
* Train one or multiple classifiers on the features.
* Implement a sliding-window technique and use the trained classifier to search for vehicles in images.
* Run the detection pipeline on a video stream and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.
* Optionally combine this project with the lane finding project


### Feature extraction

In the first step the features for the training must be extracted from the labeled data. Beside the provided dataset for the project I also used the [Udacity labeled dataset](https://github.com/udacity/self-driving-car/tree/master/annotations) 1 and 2. These datasets do not provide 64x64 labeled images but hole images with bounding boxes and corresponding labels. Therefor 64x64 images patches for each car bounding box have to be extracted. Additionally for each car bounding box I extracted an additional image patch of the same size of the bounding box which does not contain any car. This preparation happens in the file `dataset_tools.py` in the function `prepare_dataset`.


After I got all 64x64 car and non-car images the features could be extracted. I used color, histograms of color and histograms of oriented gradients as features. As the color space the YCrCb representation is used.

The whole feature extraction resulted in a feature vector of 8412 dimensions for each labeled image.

The feature extraction is done in the function `get_features` in the file `features.py`.

Here is an example of an car and an non-car image:

![alt text][car]
![alt text][non_car]

#### Color features

The color features are just the to 32x32 resized image as vector.

In the code this happens in the file `features.py` in the function `__get_color_features`.

#### Histogram of color features

For the histogram features the histogram of each color channel is taken and all histogramms concatenated. As the number of bins for the histogram I choosed 32.

This is done in `__get_hist_features`.

#### Histogram of Oriented Gradients features

For these features the histogram of oriented gradients (HOG) of the image patch is taken.
Experimenting with the parameters for the HOG lead to a decision of `'orient': 9`, `'pix_per_cell': 8` and `'cell_per_block': 2`.

The HOG happens in `__get_hog_features`.

Here is an example of the result:

![alt text][car1]
![alt text][car2]
![alt text][car3]
![alt text][car4]
![alt text][non_car1]
![alt text][non_car2]
![alt text][non_car3]
![alt text][non_car4]

![alt text][car1_hog]
![alt text][car2_hog]
![alt text][car3_hog]
![alt text][car4_hog]
![alt text][non_car1_hog]
![alt text][non_car2_hog]
![alt text][non_car3_hog]
![alt text][non_car4_hog]

### Data preparation

#### Training and test set

For training and testing the labeled data must be splitted into a training and test set. Because the labeled images are taken from a video they can't just be shuffled and splitted. This could result in nearly the same images appear in both sets and therefor the test data is learned in the training.

To avoid this I took the beginning and the ending of the video as test set and the middle part as training set.

This happens in the function `get_data` in `dataset_tools.py`

#### Feature preparation

Just using the extracted features as they are will lead to a bad result. Therefor the mean over all features is removed and then the features are scaled to unit variance. This is done by fitting a `StandardScaler` on the training data and then transform the training and test data using it.

Next to scaling the features different other preparation steps could be applied.
For example a SVM with a nonlinear kernel gets really slow on training and predicting when it is trained on a 8412 dimensions feature vector. It needed about 15 seconds to predict 1000 image patches. Because an aim was to run the pipeline in nearly real time this classifier wasn't useful. Therefor I tried reducing the feature dimension by applying a PCA on the data. Interestingly reducing the feature size to only 20 still leads to an average precision of 0.96. Sadly this was only theoretically god because it leads to way too many false positives in the video pipeline, so I dismissed the usage of PCA and just used all features.


### Classifier training

After getting the features I trained a classifier on them. I tried to find a good classifier using a support vector machine and a grid search over its parameters. Therefor I used the `SVC` class of sklearn. While the resulting classifier was pretty good it still wasn't good enough for the video pipeline. As mentioned before it was way too slow.

Using a decision tree instead leaded to a very fast classifier but only with a average precision of 0.94.

In the end it turned out that using `SVC` class was a bad idea. Even when using it only with a nonlinear kernel it was slow. Using the `LinearSVC` instead resulted in a fast an reliable classifier. The SVM is trained `__create_svm` function in the file `classifier.py`

As an additional classifier I trained a AdaBoost meta classifier with an ensemble of 50 decision trees. This is done in the function `__create_dt`.

The SVM was slightly faster than the decision trees, so I used the decision trees to verify the results from the SVM. When the SVM detects a car in an image patch the same patch is labeled again with the decision tree. If both classifiers label the patch as a car the probability that it really is a car is pretty high. This reduces false positives from the SVM.

Here are the results for the test errors:

|         | Precision SVM | Precision DT |
|:-------:|:-------------:|:-------------:|
| non-car | 0.96          | 0.96          |
| car     | 0.99          | 1.00          |
| average | 0.98          | 0.98          |



### Sliding window search

To get image patches from image which should be labeled I used the sliding window technique. For this I took a frame, cut out a region of interest, calculated its HOG and then slide a window over it. For each window the features are then generated like in the feature extraction.

This is done in the scales 1 and 1.8, so that I can detect cars in different sizes.

In the code this happens in the function `get_features_frame` in the file `features.py`.


### Example images of the pipeline

Here are some examples of the test images. All image patched, detected as cars, are drawn as bounding boxes in the lower left image. For each bounding box heat is added to a heatmap. this heatmap could be seen in the lower middle image. After that the heatmap is thresholded and `scipy.ndimage.measurements.label()` is used to label the thresholded heatmap.

Here are the results:
![alt text][test_image1]
![alt text][test_image2]
![alt text][test_image5]

As one can see in the last example there is a false positive in the middle of the street (lower left image). Because of the thresholding of the heatmap this false positive does not appear in the final image.


### Video pipeline

For the video pipeline the same as for the single images is done. Additionally to that not for every frame a new heatmap is taken, but the heatmap from the previous frame multiplied with 0.7.
Doing this results in an weighted average heatmap, which helps against false positives or false negatives in single frames.

The frame classification is done in the method `classify_frame` in the file `classification.py`

The video pipeline runs with about 6 frames per second on my laptop.

Here is [link to my video result](./output_videos/project_video.mp4)



### Combination with lane finding

This project could be combined with the last one. The lane finding can be added to the video by running each frame through the pipeline in the method `process_image` in `P4.py` from project 4.

[This](./output_videos/combi_project_video.mp4) is the resulting video.


### Discussion

The pipeline I used works well on the provided video. It can label about 6 frames per second. getting it faster than that was really hard, because the feature extraction took the most of the time. Using an approach which uses less features might result in a faster pipeline.

An other point is that the bounding boxes are not perfectly stable and there is no confident score for each detected box. Using a neuronal net might help to tackle these problems.
