# Deep Learning

## 1. Introduction

### 1.1 What is Deep Learning?

Deep Learning is a sub-field of machine learning that gets inspiration from how the brain works and tries to "simulate" its neural networks.
The NNs used in deep learning are called Artificial Neural Networks (ANNs) since they are not an actual representation of the biological neural networks, but merely share some characteristics with them, this is because we as humans have always gained inspiration from nature.
For example we took inspiration from birds in order to create airplanes and we took inspiration from mules and horses in order to build cars and trains. But even then none of these machines are an exact metal copy of those animals, we only took some fundamental characteristics and replicated them in the most useful why to us.

**Note:**
In ML we have different terms for referring to an Artificial Neural Network (ANN) such as *"model", "neural network", or "net"*.
## 2. Artificial Neural Networks Architecture

### 2.1 How ANNs work?

An ANN is basically a computing system that is composed of a collection of connected units called neurons which are structured into layers. These connected *"neural"* units form what we call the *"network"*.

Here's the representation of an artificial neural network:

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/ann_color.png?raw=true "Artificial Neural Network")

As we can see we have three layers:

- Input layer
- Hidden layer (or layers)
- Output layer

Each neuron in the network sends a signal to another, once this signal is received from another neuron it is then processed and sent to the other forward neurons to which is connected.

The neurons in the **input layer** represent the input data that we feed the network, while the neurons in the **output layer** represent the possible desired output, for example if we are trying to perform emotion detection (let's choose for simplicity only happiness and sadness) then we have only 2 output nodes.
The **hidden layer/s** contains an arbitrarily number of neurons for each layer, it's our goal to tweak it in order to get the desired outcome.

### 2.2 What Layers do?

As we have seen an ANN is composed of different layers, but we haven't talked about how they work. Let's first say that there are different types of layers in a neural network, and each one of them is used to compute different transformations on the input they take. Some example of types of layers are:

- Pooling Layers
- Convolutional Layers
- Dense Layers
- Recurrent Layers
- Normalization Layers

The reason we might choose one over another is based on the task we try to achieve. For example **convolutional** layers are better suited for image related tasks, while **recurrent** layers can perform better on time series data.

But how they perform this transformations? Well, has we can see in the image above, each neuron in layer can be connected to another neuron or several neurons in the next layer. Each connection from the first and second layer sends the computed output of the previous neuron to the input of the following neuron.
Every connection between two neurons has associated to it a particular **weight** which is simply a number that indicates the strength of the connection between the two neurons. This is because when the network receives an input at any given neuron and then it has to be passed to the next neuron trough a connection, this input will be multiplied by the weight assigned to the connection between the two neurons.
For every neuron in the second layer, a weighted sum is then computed with each of the incoming connections. Then this sum is passed to an activation function, which performs some type of transformation on the given sum. (There are several types of activation functions, for example **Sigmoid Activation Function**. We will later see how they work).

    neuron output = activation(weighted sum of inputs)

This process wil be applied until we reach the output layer, the entire process from input layer to output layer (for a given sample from the dataset) is called a **Forward Pass Through** the network. When all of the components in the dataset went through this forward pass, we say that an **epoch** is complete.

### 2.3 What are Activation Functions?

In an ANN an activation function is basically a function that maps a neuron's inputs to it's corresponding output.
As we described already, we take the weighted sum of each incoming connection for each neuron in our layer and then pass that weighted sum to an activation function. Normally this activation function performs some type of computation to transform (most of the time it's a **non-linear transformation**) the sum to a number that is usually bounded between a specific lower and upper limit.

To understand the need for an activation function let's first explain how linear functions work.

Assumptions:

- Assume that *f* is a function on a set *X*;
- Assume that *a* and *b* are in *X*;
- Assume that *x* is a real number.

The function *f* is said to be a linear function if and only if:

<img src="https://render.githubusercontent.com/render/math?math=f(a+b) = f(a) + f(b)">

and

<img src="https://render.githubusercontent.com/render/math?math=f(xa) = xf(a)">

The problem with using linear functions in deep neural networks is that the composition of two linear functions is also a linear function, meaning that in every DNN if we have only linear transformations on the data during a forward pass, then the learned mapping in our network from input to output would also be linear. But normally the types of mappings that we are trying to learn with our DNN are much more complex than simple linear mappings.

For this reason it's better to use activation functions in deep neural networks, since most of the activation functions are non-linear, which allows our network to compute arbitrarily complex functions.

Let's see now two of the most used types of activation functions:

- Sigmoid Activation Function
- RELU Activation Function
- Softmax Activation Function

#### Sigmoid Activation Function

Sigmoid takes in an input and does the following:

- For most negatives inputs, sigmoid will transform the input to a number very close to 0;
- For most positive inputs, sigmoid will transform the input into a number very close to 1;
- For inputs relatively close to 0, sigmoid will transform the input into some number between 0 and 1.

So 0 is the lower bound limit and 1 is the upper bound limit.

$$
  sigmoid(x) = \frac{e^x}{e^x\ +\ 1}
$$

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/sigmoid_activation.png?raw=true "Sigmoid Activation Function")

To better understand the "why" we use an activation function let's take for reference how a biological brain works.

In a biological brain different neurons are activated (fired) by the presence of different stimuli. For example a rat seeing a predator will fire the neurons responsible for "predator alert mode" and make it escape. A rat smelling the scent of a partner will have other neurons firing completely different from the ones responsible for "predator alert mode". Basically some neurons are either activated or they are not. This can be represented by either a 0 for not activated and a 1 for activated.

So with the use of the Sigmoid activation function we are trying to make out artificial neurons behave similarly to the biological ones, with values close to 0 representing low or no neuron activation and values close to 1 representing high neuron activation.

Since we defined that:

    neuron output = activation(weighted sum of inputs)

We can also define for example:

    neuron output = sigmoid(weighted sum of inputs)

Which means that each output from the neurons in a layer will be equal to the Sigmoid result of the weighted sums.

#### ReLU Activation Function

Sometimes we want the function to not transform the input to be a number strictly between 0 and 1. For example we want to make sure that the more positive a neuron is the more activated it is, so a better activation function for this type of transformation is ReLU which stands for *rectified linear unit*. ReLU transforms the input to the maximum of either 0 or the input itself.

$$
  relu(x) = max(0,x)
$$

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/relu_activation.png?raw=true "Relu Activation Function")

#### Softmax Activation Function

In ANN there are two classes of activation functions, the ones used in the hidden layers and ones used in the output layers. Normally an activation function used on a hidden layer is also used on all the other hidden layers, it's unlikely to see ReLU used on the first hidden layer, followed by a Tanh (Hyperbolic Tangent) function, it's usually ReLU or Tahn for all the hidden layers.

When it comes to the output layer we need a function that takes any values and transforms them into a probability distribution. That's the reason we use Softmax.

Softmax is great for classification problems (such as classifying facial emotion expression), especially if we're dealing with multi-class classification, since it will report back the "confidence score" for each single class.

$$
  Softmax(x_{i}) = \frac{\exp(x_i)}{\sum_j \exp(x_j)}
$$

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/softmax_activation.png?raw=true "Softmax Activation Function")

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/softmax.jpg?raw=true "Softmax Activation")

The scores returned by the softmax activation function will add up to 1 since we're dealing with probabilities. The predicted class is hence the item in the list with the highest confidence score.

## 3. Training an Artificial Neural Network

### 3.1 What it means to train a network?

The goal of training a neural network is to just try to solve an optimization problem, to more specific we are trying to optimize the weights in a model.
Our task is to find the weights that most accurately map our input data to the correct output class. This mapping is what the network must learn.
Since the each connection between neurons has an arbitrary weight assigned to it, during training these weights are iteratively updated and moved towards their optimal values.

### 3.2 Stochastic Gradient Descent (SGD)

Stochastic Gradient Descent is an optimization algorithm (optimizer) that is used to optimize the weights. There are many other optimizers but the most used one is SGD which basically tries to minimize some given function that we call **"loss function"**. In other words SGD updates the model's weights in order to make this loss function as close as possible to its minimum value.

### 3.3 Loss Function

Since we defined that optimizers try to minimize a loss function, what do we we actually intend with "loss"? Well, let's suppose we have a model which we want to train to classify whether an image contain either a hotdog or a pizza. We will supply the model with images of hotdogs and pizzas alongside the appropriate labels for each of these images that indicate if it's a hotdog or a pizza. Suppose now that we feed as input an image of a hotdog to our model, after a the completion of a forward pass the model will give us a prediction as output, which will basically consist of what the model believes the image contains, either a hotdog or a pizza.
The output is just probability values for each class, for example in our case we can have an output like:

- 92% hotdog
- 8% pizza

Meaning the model thinks there's a high chance that the image contains a hotdog, and if we think about it, the way the model makes this decision is very similar to us humans since we base our decisions on predictions.
The loss is just the *error* or difference between what the model predicts compared to the real label of the image, and the scope of SGD is to try to minimize this error in order to make our model's predictions as precise as possible.

For example let's assume we set the label values as:

- 0 &rarr; hotdog
- 1 &rarr; pizza

If we pass a picture of a hotdog to the model, and the output of it is `0.33`, then we conclude that the **error** (difference) is:

    error = 0.33 - 0.00 = 0.33

And we keep computing this error for each output. By the end of each epoch the error is accumulated across all the individual outputs.

So this "training" of the model is basically the model actually "learning".

There are different types of loss functions, such as *mean squared error* (MSE), but it's our task to decide which loss function to used based on our goal.

### 3.4 (MSE) Mean Square Error

In order to calculate the mean square error for a single input we simply square the computed error of this input:

$$
  MSE(input) = (error(input))^2
$$

But in order to calculate the MSE of a batch of inputs we have to take the average of the sum of all the squared errors:

$$
  MSE(batch) = \frac{1}{N}\sum_{i=1}^{N} (error(i))^2
$$

During the training, if we feed the training dataset to the model in batches, such as that `bacth_size=50` then the MSE must be calculated at then end of each batch training.

### 3.5 Gradient of the Loss Function

When we talk about gradient in neural network we simply intend the derivative of a function of several variables.
The gradient of the loss function is calculated based on each weight in the network after each is loss is calculated by using *backpropagation*.

### 3.6 How the network learn

As we stated, the moment we initialize the network we also set arbitrary values for the network weights. These weights will get updated in order to help reconcile the differences between the actual and predicted outcomes for subsequent forward passes. By using the output at the end of the model we can compute the loss (this loss depends on the used loss function) for that particular output by looking at how much difference there is between the actual and predicted outcome.
This process of updating the weights is what we call "learning".

### 3.7 Weights Update

In order to update the weight we have to three variables:

- Actual Weight
- Gradient
- Learning Rate

The computation to calculate the updated weight is:

    updated weight = actual weight - (gradient * learning rate)

This computation will be then applied to all the weights in the model every time data from the dataset pass through, with the difference that when the gradient of the loss function is calculated , the value for the gradient will be different for each weight since it's computed based on this weight.

### 3.8 Learning Rate

The learning rate is a hyperparameter that we normally set between `0.01` and `0.0001` and is used to represent the *step size* our model takes to update the weights in order to get closer to the minimized loss.

The higher the `learning_rate` value is, the more we can overshoot by taking larger steps towards the minimized loss function but miss it by shoot pass this minimum. So the solution is to set the `learning_rate` to a lower value at the cost of time since it will take longer to reach the minimized loss.

## 4 Datasets

When we want to create a neural network we usually create different data sets each for a particular reason. We can divide the dataset in:

```
Dataset
│
├── Training Set
│
├── Validation Set
│
└── Test Set
```

### 4.1 Training Set

The training set is the most import one since it contains all the data that we will feed our model in order to train it. The data it provides will be used by the model to train over it during each epoch. The goal is to extract the features of the data in order to be able to recognize this features later on data not contained in the training set, hence never sen data by the model.

### 4.2 Validation Set

The validation set is used to validate the model during the training phase in order to give more insight on problems such as *overfitting* or *underfitting*.

At the end of each epoch in during the training, the model will be validated by using the data in the validation set. This data will not be used to update the weights of the model, since it will otherwise compromise the use of the data itself by making the model already familiar with it, defeating so the it purpose.

### 4.3 Test Set

The data in the test set is used to check how our model accuracy is after the model has been trained fully. It's very important to test our model on this never seen and unlabeled data since we must make sure the neural network behaves as intended before deploying it.

### 4.4 Overfitting

Overfitting is a problem we incur in when our model performs very good on data contained in the training set, but doesn't perform as good on data that was not trained on. Generally this is because the model is not able to generalize well.

If we provide a validation set, then we can get insightful information such as:

- vA &rarr; validation accuracy
- vL &rarr; validation loss
- tA &rarr; training accuracy
- tL &rarr; training loss

We can say with certainty that our model is overfitting if the validation metrics are highly worse than the ones provided by the training:

    (tA > vA) && (tL < vL) = Overfitting

We can also finalize that the model is overfitting when it performs well on the training set but performs poorly on the test set.

In order to reduce overfitting we can use different solutions:

- **Adding more data to the training set** &rarr; The simplest solution is to add more data to be trained on. We should also make sure it is diverse enough so the model can generalized well.
- **Data augmentation** &rarr; If we can't add more data to the training set then we can use data augmentation techniques so "create" more data. For example if we are solving an image classification problem, by *cropping*, *flipping* or *rotating* the data in the training set we can add slightly more diversity to the set.
- **Reduce the complexity of the model** &rarr; Another solution would be to reduce the complexity of the model, by reducing the number of neurons or the number of layers.
- **Dropout** &rarr; A frequently used solution to solve the overfitting problem is by using *dropout*, which basically consists on randomly ignoring (dropping) some subset of nodes in a particular layer during the training phase. For example by setting `Dropout(0.2)` we will dropout rate to `20%`, meaning that 1 in 5 inputs will be randomly dropped.

### 4.5 Underfitting

Underfitting is even more problematic than overfitting, since the model  doesn't perform good on neither the training data not the validation or never seen data. We can deduce we have an underfitting problem when the training accuracy is low and (or) the training loss is high.

In order to reduce underfitting we can:

- **Increase model complexity** &rarr; The first solution against underfitting is by increasing the complexity of the model if the data we are training the model on is complex. We can do this by increasing the number of layers in the model, adding more neurons or changing the type of layers used.
- **Increase Features to the Data** &rarr; Adding more features to the data we feed the model for training can help reduce underfitting.
- **Reduce Dropout** &rarr; In the case of underfitting we should reduce the dropout or not use it at all.

## 5 Convolutional Neural Networks

Nowadays the most used type of artificial neural network for image processing and computer vision is the *Convolutional Neural Network*, CNN or ConvNet for short. The name comes form the fact that it has hidden layers called *convolutional layers*, which help detecting patterns in images. These convolutional layers, as any other type of layers, have inputs and outputs, that we specifically call in this case:

- *input channels*
- and *output channels*

Each convo-layer performs a transformation that we call *convolutional operation*, which mathematically is called *cross-correlation*.

### 5.1 Convolutional Layers

The convolutional layers, as any other type of layers, have inputs and outputs, that we specifically call in this case:

- *input channels*
- and *output channels*

Each convo-layer performs a transformation that we call *convolutional operation*, which mathematically is called *cross-correlation*.

### 5.2 Patterns

For each convo layer we must define the number of filters that the layers needs to have which indeed are able to detect patters. Such example of these patterns in an image could be *shapes*, *colors*, *objects*, *lines*, *edges*, *circles* and so forth.

For example a filter that can detect lines in an image is called *line detector*. Normally these simple features are picked up at the beginning of the convolutional neural network and get more complex towards the end og the network. For example it might start by picking up features such as edges and ending up detecting facial features such as eyebrows, eyes and ears. Going down the line we can even detect more complex objects such as humans, animals, cars and so on. Basically we start from simple-basic geometrical features and end up with complex real objects.

### 5.3 Kernel

In order to understand how kernels (a filter used to extract features from the image) work, we will assume we have a ConvoNet that takes as input images of handwritten numbers and tries to classify them into their respective classes (in this case form 0 to 9).

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/mnist.jpg?raw=true "MNIST Dataset")

In the first hidden convo-layer we will have to specify how the kernel size which will determine the number of output channels. The kernel is nothing but a tensor, a small matrix that we initialize with random values. In the example below we set the `kernel_size=(3,3)`.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/mnist_cnn.png?raw=true "CNN")

The `3x3` pixels kernel block will slide across the  `3x3` input channel pixels, and compute (*dot product*) new pixels for the output channel.

Suppose we define:

- *i* as a `3x3` matrix of pixel in the input channel
- *k* as a `3x3` matrix of pixels in the kernel

Such as:

$$
i = \begin{bmatrix}
i_{1,1} & i_{1,2} & i_{1,3}\\
i_{2,1} & i_{2,2} & i_{2,3}\\
i_{3,1} & i_{3,2} & i_{3,3}
\end{bmatrix}
$$

$$
k = \begin{bmatrix}
k_{1,1} & k_{1,2} & k_{1,3}\\
k_{2,1} & k_{2,2} & k_{2,3}\\
k_{3,1} & k_{3,2} & k_{3,3}
\end{bmatrix}
$$

Then the dot product (basically the sum of the pairwise products) is:

$$
i_{1,1}k_{1,1}+i_{1,2}k_{1,2}+i_{1,3}k_{1,3} + ... + i_{3,3}k_{3,3}
$$

The goal is to slide (*convolving*) across the entire input channel and create an output channel called **feature map**, which indeed will become the input channel for the successive layer in the neural network. Just as shown in the image below.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/convo_feature.jpg?raw=true "Kernel")

If we want to have an output channel with the same size as the input channel, we could use zero padding

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/zero_padding.jpg?raw=true "Zero Padding")

Since we are converting the images to grayscale then we have only a single color channel. If we were using RBG then we would use 3 color channels and we would perform the convolving for each channel.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/rgb_feature_map.png?raw=true "RGB Feature Map")

### 5.4 Feature (Patter) Detection

In order to understand how the model detects patterns and edges, we will use as input a grayscale image from the MNIST dataset.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/mnist_seven.png?raw=true "Grayscale Input")

By using `3x3` filters for the first convo-layer, we could detect very simple features such as:

- *top horizontal edges*
- *left vertical edges*
- *bottom horizontal edges*
- *right vertical edges*

To do this we will set the `3x3` filters with 3 types of values:

- **1** for representing white
- **0** for representing gray
- **-1** for representing black

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/mnist_filters.png?raw=true "Grayscale Features")

By convolving each filter over the input image, we would get the following input channels:

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/mnist_output_channels.png?raw=true "Output Channels")

As we can see the respective type of edge (shown with bright white pixels) is detected in each output channel.

This a very basic features, but in case we are performing more complex detection, such as emotion detection on human faces, then the more we go deep in the model last layers the more complex features we will be able to detect, such as full faces.

### 5.5 Max Pooling

Max pooling is a technique used in CNNs in order to:

- **Lowering Computational Cost**
- **Lowering Overfitting**

In order to understand how max pooling achieves this, let's briefly see how it works.

When we use max pooling after each convolutional layer, we reduce the pixel dimension of the final output channel for that layer. After we get the feature map by using the `3x3` kernel from previous example, we will get (in this case) the following result:

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/maxp_input.png?raw=true "26x26 Convolutional Layer Output")

Which has a pixel dimension of `26x26`.

Now for max pooling to work, we define a `pool_size=(2, 2)` which sets a `2x2` pixel region that will be used as filter for the max pooling operation. We also define `strides=(2, 2)` which will determine by how many pixels the filter should slide across the image. By setting `strides=None` we imply that it should default to `pool_size` values.

Now that `pool_size` and `strides` are set, the  `2x2` region will start from the top-left region of the convo-layer output, and calculate the max-value for that region and store it the output channel. We then move by the number of strides set and repeat the max-value operation, until we go over the entire convo-layer output. By the end (by using `pool_size=(2,2)` and `strides=(2,2)`) we will reduce the convo-layer output by a factor of 2, getting an output channel having a dimension of `13x13` pixels.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/maxp_output.png?raw=true "13x13 Output Channel")

Another variation is this kind of technique is **average pooling** where the average of the pool is taken instead of the max value.

![alt text](https://github.com/Salah-Akil/emotion-recognition/blob/main/markdown/images/maxp_avgp.png?raw=true "Max Pooling vs Average Pooling")

Now we can finally understand how max pooling helps lowering computational cost, since by reducing the convo-output dimensions the model will look at larger regions of the image going forward, hence reducing the amount of parameters in the model.
At the same time max pooling helps with overfitting by creating an output channel with the most activated pixels while discarding the pixels with lower activation values.

### 5.6 Batch Size

In order to speed up the training process we can pass the training sample to the model in batches (also known as mini-batches). The number of samples each batch contains heavily depends on the computational power at our disposal. The more computational resources we have access to, the more sample we can include in batch. But larger batch sizes can cause the model to generalize poorly, so the batch size is another hyperparameter to test and fine tune.

### 5.7 Batch Normalization

Sometimes we might have values in our dataset that are not normalized, for example suppose we have data on the net worth of teenagers with age between 14 to 18. We might have a teenager that has 10$ while another one which is the son of a billionaire might have 100.000.000$. This data values are not on the same scale, and high value data points in the dataset can create instability since they can cascade down the layers in the model, creating so imbalanced gradients and slowing down the training speed. In order to solve this problem it is important to apply batch normalization in order to normalize the data values.

### 5.8 Fine Tuning

Creating a model from nothing can be time consuming and hard since we must go trough trial and error many times in order to create a model that solves the initial task problem we had. We must tweak different aspects of the model such as:

- Number of neurons per layer
- Number of layers
- Types of layers
- Batch size
- Activation function
- Learning rate

and many other hyperparameters.

In order to speed that the training process we could *fine tune* it by using an already trained model that performs well on a particular task which is similar to the task we're trying to perform. We can take advantage of what this model already learned and apply it to our model.

For example if we are training a model to detect wolves in a photos, we could take a model that is already able to detect for example cats or dogs and fine tune it to our task. We will have to drop some of the last layers that make the final prediction whether it detect a dog or a cat since this final layers have learned to detect features that are too specific to the original task. But the initial layers that detect edges, and middle layers that detect more complex features such as legs or the tail could be very useful.
