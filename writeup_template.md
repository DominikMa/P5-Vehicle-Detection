# **Vehicle Detection Project**

This repository contains my solution for the project "Vehicle Detection Project" of the Udacity Self-Driving Car Engineer Nanodegree Program. The main python code to run the project could be found in [P5](P5.py), the generated images in the folder [output_images](output_images/) and the videos in the folder [output_videos](output_videos/).

The following part of the README contains a writeup which describes how the vehicle detection is achieved.

---

## Writeup

[//]: # (Image References)
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[video1]: ./project_video.mp4

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

In the first step the features for the training must be extracted from the labeled data. Beside the provided dataset for the project I also used the Udacity labeled dataset 1 and 2. These datasets do not provide 64x64 labeled images but hole images with bounding boxes and corresponding labels. Therefor 64x64 images patches for each car bounding box have to be extracted. Additionally for each car bounding box I extracted an additional image patch of the same size of the bounding box which does not contain any car. This preparation happens in the file `dataset_tools.py` in the function `prepare_dataset`.

After I got all 64x64 car and non-car images the features could be extracted. I used color, histograms of color and histograms of oriented gradients as features. As the color space the YCrCb representation is used.

The whole feature extraction resulted in a feature vector of 8412 dimensions for each labeled image.

The feature extraction is done in the function `get_features` in the file `features.py`.

Here is an example of an car and an non-car image:
---
[//]: # (TODO)
---

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
---
[//]: # (TODO)
---

### Data preparation

#### Training and test set

---
[//]: # (TODO)
---

#### Feature preparation

Just using the extracted features as they are will lead to a bad result. Therefor the mean over all features is removed and then the features are scaled to unit variance. This is done by fitting a `StandardScaler` on the training data and then transform the training and test data using it.

Next to scaling the features different other preparation steps could be applied.
For example a SVM with a nonlinear kernel gets really slow on training and predicting when it is trained on a 8412 dimensions feature vector. It needed about 15 seconds to predict 1000 image patches. Because an aim was to run the pipeline in nearly real time this classifier wasn't useful. Therefor I tried reducing the feature dimension by applying a PCA on the data. Interestingly reducing the feature size to only 20 still leads to an average precision of 0.96. Sadly this was only theoretically god because it leads to way too many false positives in the video pipeline, so I dismissed the usage of PCA and just used all features.


### Classifier training

After getting the features I trained a classifier on them. I tried to find a good classifier using a support vector machine and a grid search over its parameters. Therefor I used the `SVC` class of sklearn. While the resulting classifier was pretty good it still wasn't good enough for the video pipeline. As mentioned before it was way too slow.

Using a decision tree instead leaded to a very fast classifier but only with a average precision of 0.94.

In the end it turned out that using `SVC` class was a bad idea. Even when using it only with a nonlinear kernel it was slow. Using the `LinearSVC` instead resulted in a fast an reliable classifier.




Here are the results:

|         | Precision SVC | Precision DT |
|:-------:|:-------------:|:-------------:|
| non-car | 0.96          | 0.96          |
| car     | 0.99          | 0.96          |
| average | 0.98          | 0.96          |




### Sliding window search



### Video pipeline


### Combination with lane finding




#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I trained a linear SVM using...

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

I decided to search random window positions at random scales all over the image and came up with this (ok just kidding I didn't actually ;):

![alt text][image3]

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

Ultimately I searched on two scales using YCrCb 3-channel HOG features plus spatially binned color and histograms of color in the feature vector, which provided a nice result.  Here are some example images:

![alt text][image4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video.mp4)


#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

I recorded the positions of positive detections in each frame of the video.  From the positive detections I created a heatmap and then thresholded that map to identify vehicle positions.  I then used `scipy.ndimage.measurements.label()` to identify individual blobs in the heatmap.  I then assumed each blob corresponded to a vehicle.  I constructed bounding boxes to cover the area of each blob detected.  

Here's an example result showing the heatmap from a series of frames of video, the result of `scipy.ndimage.measurements.label()` and the bounding boxes then overlaid on the last frame of video:

### Here are six frames and their corresponding heatmaps:

![alt text][image5]

### Here is the output of `scipy.ndimage.measurements.label()` on the integrated heatmap from all six frames:
![alt text][image6]

### Here the resulting bounding boxes are drawn onto the last frame in the series:
![alt text][image7]



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

Here I'll talk about the approach I took, what techniques I used, what worked and why, where the pipeline might fail and how I might improve it if I were going to pursue this project further.  
