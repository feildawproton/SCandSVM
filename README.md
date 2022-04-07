# SCandSVM
An old (2016) implementation of sparse coding and SVM.  This was applied to iMEMS data for gesture classification. 

The report below was from school (MSIM 607) and describes these techniques applied to imagery.

# Comparing the Effects of Training and Encoding: with Random
Patches, Vector Quantization, and Sparse Coding Dictionary
Training and Sparse Coding and Soft Threshold Encoding

## Abstract
In this experiment we used the learning
framework from Coates et al on a data set
consisting of skin cancer images and non-skin
cancer images. We tested combinations of
Vector Quantization, Sparse Coding, and
Random Patches dictionary training algorithms
with Sparse Coding and Soft Threshold encoding
algorithms. With this data set we saw greater
performance from Soft Thresholding than
Sparse Coding encoding, particularly at smaller
dictionary sizes (100 and 300). For the training
portion we saw the Orthogonal Matching
Pursuit Vector Quantization algorithm
underperform the other training algorithms. Of
particular note the efficient combination of
Random Patches, Soft Thresholding, and the
smallest dictionary size we tested performed as
well as or better than any other combination
(83.33% testing accuracy).
The author also tested a trail batch of depth
images of a human performing various
exercises. The data set was not broadly varied,
however the trial is promising as all
combinations were able to classify 5 separate
exercises, across 732 images, at 100% accuracy
(using the same settings as the skin cancer
data).
## 1. Introduction
In recent years new powerful algorithms, such
as Sparse Coding, have been consistently
yielded better results than older mainstays,
such as Vector Quantization, for generating
image features. In their work Coates et al seeks
to determine the feature learning contributions
from dictionary training and encoding.
Dictionary training learns a set of basis
functions, while encoding defines a mapping
between a data vector x and a feature vector f.
Coates et al tested combinations of 5 dictionary
training algorithms and 4 encoding algorithms.
Their results suggest that good encoding has a
greater performance affect than dictionary
training. They also show that a simple encoder
such as Soft Thresholding can be competitive
with Sparse Coding.
In this experiment we combine 3 dictionary
training algorithms and 2 encoding algorithms.
Because the two encoding algorithms we used
were Sparse Coding and Soft Threshold, and
since Soft Thresholding is competitive with
Sparse Coding, the influence of encoding is
somewhat muted though noticeable. We do
note that with good encoding a less expensive
dictionary learning algorithm can be used to
good effect.
## 2. Methods
We use the learning framework from Coates et
al, 2011. However, we did not extract image
data and class labels from batches (such as
CIFAR-10). Instead, we read in the images
directly and their class labels come from the
folder they were in. 
### 2.1 Image Loading
Since we don’t have a testing batch, we use a
user defined training fraction to separate our
training and testing data. In the experiments
we conducted 5/6 of the data was used for
training and 1/6 was used for testing. This
follows the same ratio that Coates et al. used
with the CIFAR-10 data.
Upon loading into memory, the images were
compressed to a format of 100x100x1. That is
they were compressed to 100 pixels by 100
pixels with 1 byte used for grayscale intensity.
The work we are basing these experiments off
of used 3 bytes for color.
After loading and compression the images are
concatenated into a single three dimensional
matrix, and the class labels are concatenated
into a two dimensional matrix. It should be
noted that the images keep their dimensionality
in this format (i.e. their pixel values were not
formatted into a single vector), and care should
be taken if duplicating these experiments with
image data whose dimensionality has been
reduced.
### 2.2 Patch Creation
We use image patches for creating the
dictionary. The image patches are randomly
selected from the entire subset of images.
Therefore it is unnecessary to randomly select
from a subset of batches. The image patches
are 6x6 pixels and are normalized in the same
manner provided by Coates et al, however
changes were made due to the dimensionality
differences in the data sets.
### 2.3 Dictionary Creation
After patch creation the dictionary is then
created. In these experiments the dictionary
was trained with three algorithms: Random
Patches, Orthogonal Matching Pursuit for
Vector Quantization, and Sparse Coding.
##### 2.3.1 Random Patches
This training algorithm fills the columns of the
dictionary with randomly sampled and
normalized vectors from the training data set.
#### 2.3.2 Orthogonal Matching Pursuit
This algorithm uses an alternating minimization:
𝑚𝑖𝑛
𝐷,𝑠(𝑖)
∑ ||𝐷𝑠
𝑖 − 𝑥
𝑖
||
2
2
(1)

Subject to
||𝐷
𝑗
||
2
2
= 1 (2)
|(|𝑠^𝑖 |)|0 ≤ 𝑘 (3)
Where D is the dictionary and subject using
normalized dictionary elements (2). s is the
dictionary and is subject to having no more than
k non-zeros (3). For a single input x(i) from our
patches, s(i) starts at 0, then the algorithm
selects an element of s(i) to be non-zero, and
then S(i) is update to minimize (1).

2.3.3 Sparse Coding
This is similar to OMP but the optimization
function is:
𝑚𝑖𝑛
𝐷,𝑠(𝑖)
∑ ||𝐷𝑠
𝑖 − 𝑥
𝑖
||
2
2
+ 𝜆||𝑠
𝑖
||1 (4)

Subject to the constraint (2).
It should be noted that the reference paper
(Coates et al) also uses Sparse RBMS, Sparse
auto-encoders, and Random Weights for
dictionary training.
### 2.4 Feature Extraction
After the dictionary is created we need to
extract features from a new input vector of all
the patches. We are projecting all the image
patches x onto the dictionary D (that was
created with a subset of x) in order to create
the vector of features f.
In these experiments each choice of encoder
was paired with each training algorithm. We
used two encoding algorithms for feature
extraction: Sparse Coding and Soft Threshold
#### 2.4.1 Sparse Coding
Here we minimize (4) with D fixed and solving
for s. The positive and negative portions of s
are split into separate features:
𝑗 = max{0, 𝑠}
𝑓𝑗+𝑑 = max{0, −𝑠}
(5)

#### 2.4.2 Soft Threshold
Here the features are split via:
𝑓𝑗 = max{0,𝐷
(𝑗)𝑇𝑥 − 𝛼}
𝑓𝑗+𝑑 = max{0, −𝐷
(𝑗)𝑇𝑥 − 𝛼}
It should be noted that the reference paper also
used OMP and a “Natural” encoding.
### 2.5 SVM Training
After features f are extracted a support vector
machine (SVM) is then trained on these feature.
“minFunc” is used here. When the directional
derivative is below or the change in the
functional value is lower than a user supplied
constant the SVM stops training and reports
training accuracy.
### 2.6 Testing
After SVM training, the testing data is
concatenated and features are extracted with
the specified encoder. Testing accuracy is then
reported out.
## 3 Experiments
### 3.1 Patches
In the reference paper during dictionary
training, 400,000 patches were used for OMP,
500,000 for Sparse Coding, and 50,000 for
random patches. In these experiments the
number of patches was reduced by one order of
magnitude to 40,000, 50,000, and 5,000
respectively. This was originally done to speed
up testing, but due to the consistency of results
these values were retained
### 3.2 Training Iterations
The number of training iterations was kept the
same as the reference work. Namely 50
iterations for OMP and 10 iterations for Sparse
Coding.
### 3.3 Dictionary Size
In these experiments the dictionary size was
varied for every combination of dictionary
training and encoder. The dictionary sizes used
were 100, 300, and 1000.
### 3.4 Training and Encoder Combinations
Every combination of the three training
algorithms and two encoding algorithms was
used in this experimental design. This gives a
3x2x3 experiment:
Table 1: Experimental Design
Testing Accuracy
Encoder
Dictionary
Training
Soft
Threshold
Sparse
Coding Size
OMP VQ
Test
Accuracy
Test
Accuracy 100
Test
Accuracy
Test
Accuracy 300
Test
Accuracy
Test
Accuracy 1000
Random
Test
Accuracy
Test
Accuracy 100
Test
Accuracy
Test
Accuracy 300
Test
Accuracy
Test
Accuracy 1000
Sparse
Coding
Test
Accuracy
Test
Accuracy 100
Test
Accuracy
Test
Accuracy 300
Test
Accuracy
Test
Accuracy 1000

### 3.5 Skin Cancer Data Set
The Skin_Cancer data set consists of 288 color
images. There are two classes of images:
“Cancer” and “Non_Cancer”. There are 204
images of Non_Cancer and 84 Cancer Images.
These images were all resized to 100x100 pixels
and converted to grayscale (100x100x1). These
images are sensitive so no samples are shown
here.
### 3.6 Trial Depth Images
The author also wanted to check the feasibility
of depth image classification so a trial sample of
images was used. These images are not
validated for uniqueness. However, they are of
interest to the author and are provided here
along with results for the reader’s interest.
There are five image types: human facing away
from the camera performing a shoulder test
(Back), human facing the camera (Forward),
human performing a leg raise (LegRaise),
human performing a pushup (PushUP), and
human performing a rotary stability task
(Rotary). There are 157 Back images, 144
Forward images, 146 LegRaise images, 140
PushUp images, and 145 Rotary images. Here
are some examples (resized to 100x100 pixels):
Figure 1: Back
Figure 2: Forward
Figure 3: LegRaise
Figure 4: PushUp
Figure 5: Rotary
## 4 Results
All training results were above 95% when SVM
training was concluded because either the
directional derivative was below threshold or
the change in function value was less than
threshold. Therefore, training accuracy is not
reported here, though they are recorded in the
supplemental documents.
### 4.1 Skin Cancer Testing Accuracy
Testing accuracy results for the Skin Cancer data
is shown in the table below.
Table 2: Testing Accuracy for the Skin Cancer Data
Skin Cancer Data Testing Accuracy
Encoder
Dictionary
Training
Soft
Threshold
Sparse
Coding
Dictionary
Size
OMP VQ,
40000
Patches, 50
Iterations
68.75 72.91667 100
75 72.91667 300
75 77.08333 1000
Random,
5000
Iterations
83.33333 79.16667 100
79.16667 79.16667 300
81.25 81.25 1000
Sparse
Coding,
50000
Patches, 10
Iterations
83.33333 77.08333 100
83.33333 75 300
81.25 75 1000
Figure 6: Each Combination's Accuracy over Dictionary Size
Figure 7: Just Training Algorithm Accuracy over Dictionary
Size
Figure 8: Just Encoding Algorithm Accuracy over Dictionary
Size
### 4.2 Trial Depth Image Testing Accuracy
All combinations of training algorithm, encoder,
and dictionary size achieved the same testing
accuracy of 100%. This suggests that the data
set was not broad enough and/or we are not
near the lower limit of patches or dictionary
size.
## 5 Conclusions
Unless specified the conclusions reached here
are based on the results from the skin cancer
data. 
All testing accuracy results were close. At a
dictionary size of only 100 the combination of
Random Patches and Soft Threshold performed
better or just as good as any other combination.
Only the OMP Training algorithm showed
marked improvements with the increase in
dictionary size for the Skin Cancer Data. Soft
Threshold encoding outperformed Sparse
Coding. Also, the random patches training
algorithm outperformed or performed as well
as any other algorithm.
The combination and Sparse Coding Training
and Soft Threshold encoding performed as well
as or better than any other combination.
However it was matched by or followed closely
by Random Patches and Soft Threshold. The
interesting result from this experiment is that
the skin cancer images, less expensive training
and encoding algorithms with smaller dictionary
sizes performed just as well or better than
sparse coding.
The results for the depth image classification
has encouraged the author to run more
experiments in the lab where he works. The
reason the author would like to be able to
classify these images is because the commercial
sensor and library (Microsoft Kinect) he uses
was only trained on individuals facing the
camera. The author needs to be able to detect
individuals on the ground and facing away from
the camera while performing different physical
tasks.
## References
Coates, A., Ng, A., “The Importance of Encoding
Versus Training with Sparse Coding and
Vector Quantization” Proceedings of the
28th Annual Conference on Machine
Learning 2011
Krizhevsky, A. “learning Multiple Layers of
Features from Tiny Images” 2009
Mairal, J., Bach, F., Ponce, J. “Sparse Modeling
for Image and Vision Processing”
Foundations and Trends in Computer
Graphics and Vision Vol. 8 No. 2-3
(2012) 85-283
Schmidt, Mark “minFunc”
http://www.cs.ubc.ca/~schmidtm/Soft
ware/minFunc.html
