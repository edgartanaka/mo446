# Generating synthetic data with GMM and PCA
In this tutorial, we are briefly going to explain what a generative model is and then
show 2 practical examples of how to use Gaussian Mixture Models (GMM) to fit data and later generate synthetic data.

This tutorial has no ambition to go deep into the math details of machine learning concepts. Instead, 
we want to explore the possibilities of using GMMs in learning a data distribution.

## Concepts
In this section, we are going to define some concepts used throughout this tutorial.

### Classification
In Machine Learning, the task of classification can be defined as: given a sample of data X, predict its class Y from a finite 
set of known classes. Let's put this into more practical terms. Assume we're building a classifier to predict if
a person has a flu. A data sample could be a list of characteristics of a certain patient: 
- age
- temperature
- does s/he have a headache?
- does s/he have sore throat?
- does s/he have chills?

The classes would be simply:
- has a flu
- does not have a flu

We can think of all sorts of types of data: images, voice, text, numbers. We can also think of different types of classes
that are not necessarily binary such as types of objects in an image. 

### Generative Model vs Discriminative Model
In statistical classification, models can be divided into two groups: discriminative or generative. 

A generative algorithm models how the data was generated in order to categorize a signal. 
It asks the question: based on my generation assumptions, which category is most likely to generate this signal? 
A discriminative algorithm does not care about how the data was generated, it simply categorizes a given signal. 

One of the advantages of generative algorithms is that you can use the learned data distribution 
to generate new data similar to existing data. On the other hand, discriminative algorithms generally 
give better performance in classification tasks. (from wikipedia)

In this tutorial, we are going to focus on Generative models.

### Gaussian Mixture Model (GMM)
Gaussian Mixture Models are part of the generative models group. They try to model data distribution as a 
set of gaussian distributions as depicted below:

![GMM Example](readme/gmm.png?raw=true "GMM Example")

The number of gaussian distributions (also known as number of components in the sklearn library) must be defined
by the developer. In the image above, we have 2 gaussian distributions.

### Principal component analysis (PCA) 
PCA is a method used for dimensionality reduction. Take a look at the 2D data points in the image below:

![PCA Example](readme/pca.png?raw=true "PCA Example")

We can note two things here. First is the fact that the dimensions are somewhat correlated as the data points
lie close to an imaginary inclined line in the plot chart. The second is that the data points are more spread apart in 
one direction than in the other (take a look at the two arrows). In other words, one direction has more variance than
the other. Imagine now that we have changed the axis of data points to those marked as the black arrows and we
want to reduce that 2D space into a 1D space. If we remove the axis denoted by the shorter arrow (the one with
lower variance), we would lose some information but we wouldn't lose so much. That's what PCA allows us to do. 
If finds these new axis and allows us to pick the ones with more variance which are also the ones that carry
more information. 

In this tutorial, we are going to use the well known MNIST dataset. Each data sample in this dataset is
a square image of handwritten digits:

![PCA Example](readme/mnist.png?raw=true "PCA Example")

You can imagine that some pixels close to the borders will always be black. Those pixels don't carry any useful
information (as they have zero variance) and if removed, will not cause much information loss.

The math behind PCA uses covariance matrices and will not be covered in this tutorial.

## Tutorial
### How to run
- have docker pre-installed
- run the following command:
```
docker run -it --rm --env DISPLAY=$DISPLAY --volume="/tmp/.X11-unix:/tmp/.X11-unix:rw" -v=$(pwd)/..:$(pwd)/.. -w=$(pwd) adnrv/opencv make build test
```

This is what you should see in your terminal:
```
Running pipeline for Fashion MNIST dataset...
Saved plot to ./results/fashion_original_data.png
Original space - number of dimensions:784
Reduced space - number of dimensions:220
Saved plot to ./results/fashion_generated_data.png
FINISHED

Running pipeline for MNIST dataset...
Saved plot to ./results/digits_original_data.png
Original space - number of dimensions:64
Reduced space - number of dimensions:41
Saved plot to ./results/digits_generated_data.png
FINISHED
```

### Explaining the Code
In this section, we'll explain each piece of the source code and give you an overview of the pipeline.

This tutorial runs the same pipeline for two datasets: MNIST and Fashion MNIST.
MNIST is a collection of 8x8 images of handwritten digits.
Fashion MNIST is a collection of 28x28 images of fashion products such as bags, t-shirts, pants and others.
Although both of these datasets are labeled, we did not use the labels here. We are merely interested in 
model the data distribution here and not classifying.

The first part of the pipeline is loading the data. For MNIST, we simply load it from the sklearn library.
For the Fashion MNIST dataset, we use the function load_mnist() which loads the images from a .gz file.

Having loaded the data, we select a sample of 100 datapoints and plot it just so we can have a 
sense of how the original data looks:
```python
# original data: subsample 100 images and plot them
data = np.asarray(data)
original_sample = data[:100, :]
plot_generated_data(original_sample, image_width, OUTPUT_FOLDER + prefix_file + '_original_data.png')
```
 
Next, we use PCA to reduce dimensionality. We instantiate a PCA model specifying how much
 data variance we want to keep (the parameter "var"). The fit_transform function actually performs the 
 dimensionality reduction.
```python
# reduce dimensionality
pca = PCA(var, whiten=True)
data = pca.fit_transform(data)
```

For performance sake, we are subsampling the data for the Fashion MNIST dataset. We are not 
subsampling the MNIST dataset.
```python
# sub sample data
np.random.shuffle(data)
if sub_sample_size > 0:
    data = data[0:sub_sample_size, :]
```

Now it's time to fit the data using GMM. This is equivalent to saying that GMM is learning the data distribution.
We are using the "full" covariance type which is the least constrained option of GMM.
The function fit() adjusts the mixture of gaussians 
to the data passed. 
```python
# fit data
# this is similar to learning the gaussian distributions or learning how data is generated
gmm = GaussianMixture(n_components, covariance_type='full', random_state=0)
gmm.fit(data)
```

Finally, we create 100 synthetic data points based on the GMM that we just fitted. 
Remember that GMM learned the data distribution in that reduced space of less dimensions so
we also have to run the inverse of PCA to tranform those 100 data points back to the
original space (with 8x8 or 28x28 dimensions).
The last line plots the generated data.
```python
# generate a sample of 100 synthetic data points
X_sample, y_sample = gmm.sample(100)

# transform from reduced dimension space back to original space
generated_data = pca.inverse_transform(X_sample)

# plot synthetic data
plot_generated_data(generated_data, image_width, OUTPUT_FOLDER + prefix_file + '_generated_data.png')
```

### Results
In this section, I will go over some of the results obtained with this tutorial. 

First of all, let's talk about the dimensionality reduction with PCA. 
For MNIST, we kept 99% variance and this reduced 64 dimensions to 41 dimensions. 
For Fashion MNIST, we kept 96% variance and this reduced 784 dimensions to 220 dimensions. 
Reducing the dimensionality was a must as GMM did not work with the high dimensionalities. Sklearn was killing
 the GMM fit process if we had more than 220 dimensions in the Fashion MNIST dataset.
 
Dataset |  Original Dimensionality |  Reduced Dimensionality | Variance Kept
:-------------------------:|:-------------------------:|:-------------------------:|:-------------------------:
MNIST | 64 | 41 | 99%
Fashion MNIST | 784 | 220 | 96%
 
Let's now discuss the generated data in comparison with the original data. 
For MNIST, the results are very satisfactory. We can clearly interpret what the image is and distinguish what digits were 
written. In this case, it seems that gaussian distributions fitted the original data well and that there is not
much overlap between the gaussians (which would mean that the data distribution is more confusing). Also, we kept a
very good variance of the data: 99%. Take a look at the results:

Original Data             |  Generated Data
:-------------------------:|:-------------------------:
![GMM Example](results/digits_original_data.png?raw=true "GMM Example")  |  ![GMM Example](results/digits_generated_data.png?raw=true "GMM Example")

For Fashion MNIST, we can also distinguish what some of the objects are like shoes, t-shirts and pants. However,
some of the objects are a bit fuzzy. We can point a few reasons for that such as: 
- we may have lost too much information
with PCA
- the gaussians fitted are not well separated (i.e. data is more mixed)

Take a look at the results:

Original Data             |  Generated Data
:-------------------------:|:-------------------------:
![GMM Example](results/fashion_original_data.png?raw=true "GMM Example")  |  ![GMM Example](results/fashion_generated_data.png?raw=true "GMM Example")

Running this tutorial should give you an idea of how GMM works and what it's capable of doing. 
You should also get a grasp of how PCA works and how useful it can be in this and in other applications.


## References
- Fashion MNIST dataset: https://github.com/zalandoresearch/fashion-mnist
- Tutorial explaining GMM: https://jakevdp.github.io/PythonDataScienceHandbook/05.12-gaussian-mixtures.html
- SKLearn page for GMM: http://scikit-learn.org/stable/modules/mixture.html
- Wikipedia page for Generative Models: https://en.wikipedia.org/wiki/Generative_model
- Wikipedia page for PCA: https://en.wikipedia.org/wiki/Principal_component_analysis
