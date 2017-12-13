## Writeup

---

**Vehicle Detection Project**

The goals / steps of this project are the following:

* Perform a Histogram of Oriented Gradients (HOG) feature extraction on a labeled training set of images and train a classifier Linear SVM classifier
* Optionally, you can also apply a color transform and append binned color features, as well as histograms of color, to your HOG feature vector. 
* Note: for those first two steps don't forget to normalize your features and randomize a selection for training and testing.
* Implement a sliding-window technique and use your trained classifier to search for vehicles in images.
* Run your pipeline on a video stream (start with the test_video.mp4 and later implement on full project_video.mp4) and create a heat map of recurring detections frame by frame to reject outliers and follow detected vehicles.
* Estimate a bounding box for vehicles detected.

[//]: # "Image References"
[image1]: ./examples/car_not_car.png
[image2]: ./examples/HOG_example.jpg
[image3]: ./examples/sliding_windows.jpg
[image4]: ./examples/sliding_window.jpg
[image5]: ./examples/bboxes_and_heat.png
[image6]: ./examples/labels_map.png
[image7]: ./examples/output_bboxes.png
[img1]: ./output_images/test1.jpg
[img2]: ./output_images/test2.jpg
[img4]: ./output_images/test4.jpg
[video1]: ./project_video_output.mp4

## [Rubric](https://review.udacity.com/#!/rubrics/513/view) Points

### Writeup / README

#### 1. Provide a Writeup / README that includes all the rubric points and how you addressed each one.  You can submit your writeup as markdown or pdf.  [Here](https://github.com/udacity/CarND-Vehicle-Detection/blob/master/writeup_template.md) is a template writeup for this project you can use as a guide and a starting point.  

You're reading it!

### Histogram of Oriented Gradients (HOG)

#### 1. Explain how (and identify where in your code) you extracted HOG features from the training images.

The code for this step is contained in the first code cell of the IPython notebook (or in lines # through # of the file called `some_file.py`).  

I started by reading in all the `vehicle` and `non-vehicle` images.  Here is an example of one of each of the `vehicle` and `non-vehicle` classes:

![alt text][image1]

To extract HOG features, I used a `get_hog_features()` in `util.py`, which is
closely based on the example given in the lesson. It wraps SciKit Image's
`skimage.feature.hog` function to reduce the number of parameters required. Like
the lessons, I chose to use the `YCrCb` colorspace for this project. Because of
the way `YCrCb` splits color channels, it separates luminance from color, which
makes it good for feature detection. I chose to use all of the color channels
for HOG feature extraction.


![alt text][image2]

#### 2. Explain how you settled on your final choice of HOG parameters.

Based on the examples in the lessons, I chose to use a combination of HOG and
color features. I felt that spatial features were redundant with HOG and also
likely to introduce overfitting to the specific images in the training set.

For the other HOG parameters (orientations, pixels_per_cell, cells_per_block), I
initially tested the values used in the lesson, and I found that they worked
well (~98% accuracy), so I decided not to change them.

Here are the parameters that I ultimately used for feature extraction:

| Parameter            | Value   |
| -------------------- | ------- |
| Color Space          | YCrCb   |
| Histogram Bins       | 32      |
| Hog Orientations     | 9       |
| Hog Pixels Per Cell  | 8       |
| Hog Cells Per Block  | 2       |
| Hog Channel          | All     |
| Spatial Feature Bins | 32 x 32 |

My feature extraction methods can be found in `util.py`.

#### 3. Describe how (and identify where in your code) you trained a classifier using your selected HOG features (and color features if you used them).

I chose to use a linear Support Vector Machine (SVM). Various
background research has shown that SVM is a good compliment to HOG features

I trained my classifier from the Udacity training sets of
[vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/vehicles.zip) and [non-vehicles](https://s3.amazonaws.com/udacity-sdc/Vehicle_Tracking/non-vehicles.zip) using the included 2nd code block in the jupyter notebook, I was able
to consistently achieve accuracy of about 99.15% on a test set consisting of a
randomly selected 20% of the labeled data input. To avoid
having to retrain my classifier every time I run the program, I saved the
classifier, parameters, and scaler to a Python pickle `classifier.p` to quickly
load into the detection pipeline.

### Sliding Window Search

#### 1. Describe how (and identify where in your code) you implemented a sliding window search.  How did you decide what scales to search and how much to overlap windows?

The processing pipeline is invoked from `the 3rd code block in jupyter`, but the processing
code is in `detector.py`, which defines a `Detector` class. `Detector` is a
class object that takes in an image and returns bounding boxes of probably vehicles
using a three step pipeline:

1. At multiple scales and ROIs, perform a sliding window search using the
  feature extractor and classifier.
2. Draw the classifier "hits" (likely vehicle bounding boxes) onto a heatmap.
3. Threshold the heatmap to reduce false-positives.
4. Return bounding boxes of the thresholded heat map.

My pipeline then draws the bounding boxes from the `Detector` on the
image and displays or saves it as requested. To find cars in images, I initially used a sliding window search at multiple scales. 
I generated a set of windows using the methods in `util.py`, but I found that this 
was too slow. Instead, I pre-computed the HOG image at each scale and slid the window over the HOG image for faster
processing. Similarly, I binned the image at each scale for spatial features,
although spatial feature calculation time is minimal compared to the HOG and
histogram calculation. For the histogram calculateion, I used
`skimage.filters.rank.windowed_histogram` to rapidly compute a sliding window
histogram of the whole image at multiple scales. With these methods and well-
chosen ROIs, I was able to process frames at about one every 2.4 seconds. A lot
of the processing time is spent on a single scale--the 48 pixel size. This is a
small patch that searches over a large area with a large overlap. That single scale makes up about 1 second out of the 2.4 second processing time, but without it, I was unable to detect the car in test image 3. For other scales, I balanced
my overlap and scales to remain accurate while keeping processing time
manageable. For small scales, I reduced the ROI and overlap to reduce the
number of windows. For larger scales I was able to use larger ROIs and more
overlap. The other scales I processed each took about 0.3 seconds per frame.

Ultimately, I decided on the following set of scales, overlaps, and ROIs:

| Window Size | Overlap | ROI (x)    | ROI (y)    |
| ----------- | ------- | ---------- | ---------- |
| 32 x 32     | 0.0     | [320, 960) | [396, 460) |
| 48 x 48     | 0.5     | [0, 1280)  | [360, 540) |
| 64 x 64 *   | 0.5     | [426, 853) | [396, 648) |
| 112 x 112   | 0.75    | [0, 1280)  | [360, 630) |
| 128 x 128   | 0.75    | [0, 1280)  | [360, 630) |

#### 2. Show some examples of test images to demonstrate how your pipeline is working.  What did you do to optimize the performance of your classifier?

I took three steps to improve the reliability of the classifier. The first was
that instead of using the prediction of the linear SVM, I used the distance of
the feature vector from the decision boundary as a measure of "car-ness".
Measurements with a distance of less than 0.3 were discarded as being too close
to the borderline.

**Two cars being detected:**
![alt text][img1]

**No car is detected but a border window shows:**
![alt text][img2]

**Two cars detected:**
![alt text][img4]
---

### Video Implementation

#### 1. Provide a link to your final video output.  Your pipeline should perform reasonably well on the entire project video (somewhat wobbly or unstable bounding boxes are ok as long as you are identifying the vehicles most of the time with minimal false positives.)
Here's a [link to my video result](./project_video_output.mp4)

#### 2. Describe how (and identify where in your code) you implemented some kind of filter for false positives and some method for combining overlapping bounding boxes.

You can go through the `__call__` function in the detector.py file.

As said above I took three steps in improving my classifier. This greatly reduced the false positives without an
appreciable loss of true positive classifications. Next, I combined the detected
windows into a heatmap. Initially, I simply incremented pixels inside the
detection window, as demonstrated in the lesson. However, I found that this
method overweights large vehicles, because they tend to have redundant
detections at multiple scales. To reduce this, I weighted the detection windows
in the heatmap by a factor inversely proportional to the area of the window.
This way a large detection window has less influence on the heat map per pixel
than a small detection window. I used a simple threshold of 2.25 to
differentiate between "car" and "not-car" pixels, then used
`scipy.ndimage.measurements.label` to group continuous sets of "car" pixels and
drew bounding boxes around each. Finally, I rejected any bounding boxes smaller
than 32x32 pixels, which could sometimes form as a result of partially-
overlapping false positives. Using these methods, I was able to eliminate
false positives on the test images, although some detection bounding boxes
ended up being smaller than the cars they represented



---

### Discussion

#### 1. Briefly discuss any problems / issues you faced in your implementation of this project.  Where will your pipeline likely fail?  What could you do to make it more robust?

The two most serious issues I faced with this task are the actual shape of the output bounding boxes and the slowness of the search.

Many of the bounding boxes do not fully enclose the car they represent. It may be possible to improve the bounding boxes by reducing the heatmap threshold, but doing so often reintroduces false positives. It would be more desirable to have a classifier with fewer false positives and to rely less on heatmaps for reliability. I did not have time to explore other classifiers (or other features, such as Haar-like wavelet features) to see if any are more accurate at identifying cars. Current research suggests that a deep convolutional neural network approach would be the most effective, but this has the downside of requiring a GPU to operate at reasonable speed.

Currently, the speed of this method is very slow. Reimplementing this algorithm in C++ would probably yield an appreciable speed-up, and it may be possible to reduce the window size from 64 x 64 to 32 x 32, which would probably speed up the HOG and histogram calculate considerably. Who knows? It may even be possible to achieve the desired speed using Tensor Flow.

Overall accuracy does not seem to be a major problem for this network. Changes in lighting and shadow, which are often vexing for image processing, don't seem to have a large effect. However, it's quite good to notice that that there are no colored cars in any of the test images or video, and those might have very different feature responses.