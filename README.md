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
