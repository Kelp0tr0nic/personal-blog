---
title: Lesson 4
author: Kalpesh Chavan
description: Lecture notes converted from Jupyter notebooks.
pubDatetime: 2025-11-13T06:55:49Z
modDatetime:
draft: true
tags:
  - notebook
  - math
  - ml
---
## Recap of Previous Lessons

In the previous 3 lessons we covered various topics that helped establish the conceptual and mathematical foundations of machine learning in general. This included a basic introduction to linear regression, understanding of the techniques used to evaluate the success and performance of machine learning models, and a rigorous analysis of the math behind classification and the corresponding logistic regression algorithm.

- The mathematical foundations (linear algebra, statistic, calculus) for ML ⛏️
- An introduction to basic linear regression and polynomial linear regression 📈
- The math and difference between classification and regression.

In this lesson we will (finally) expand these concepts into the realm of deep learning.

- The conceptual objective of neural networks.
- The building blocks: Perceptrons.
- Various implementations (Pytorch vs Tensorflow w/ optional Keras API)
- Classical neural networks: Multi-layer perceptrons
- Introduction to advanced neural networks: CNNs & RNNs

## What is a Neural Network? 🧠

A neural network can be described as a graph, or a collection of nodes (or *neurons*) connected to each other by edges. Data enters the neural network through one side and exits through the other side. *Such networks are often called "feed-forward"* networks.

<img src="/notebooks/media/basic_nn.jpg" alt="a basic neural network" width="400" height="auto">

We can connect back neural networks to our original concept of a weighted sum by characterizing each node in the input layer as carrying inputs $x_1, x_2, \ldots, x_n$ and having each edge between the input and output node as the weights $w_1, w_2, \ldots, w_n$. 

<img src="/notebooks/media/basic_nn2.png" alt="a basic neural network" width="400" height="auto">

- **Logistic regression is a simple binary classifier that can be represented as a single-layer neural network. However, neural networks extend beyond logistic regression by adding hidden layers to model more complex decision boundaries.**

- **Although a basic linear regression model can be represented in a neural network, we can apply certain functions to the weighted sum to transform a linear decision boundary into a non-linear one (e.g., a curve instead of a straight line).**

- **The objective of a neural network is to classify data into 2 or more categories. Logistic regression can be thought of as the original simple binary classification problem, but neural networks deal with more complicated decision boundaries.**

Fundamentally each neuron can represent a different weight for a single input example.

- If an image has $1920 \cdot 1080 \approx 2000000$ pixels, then each of these pixels could be mapped to an input node.

- By adding more hidden layers we are attempting to abstract clusters of neurons into "average" weights with the hope that clusters of neurons can and will represent more complicated features in the image such as edges, shapes, and eventually even more abstract concepts like faces.

Lets start now by discussing the building blocks of such a network!

## Perceptrons

A perceptron can be thought of as a singular system, a single unit that applies a function.

<img src="/notebooks/media/perceptron_outline.png" width="600px">

Often times the weights are on the edges themselves and the weighted sum and activation function are grouped into the same "neuron".

So the following image is a more simplified view of a perceptron:

<img src="/notebooks/media/perceptron_outline_simplified.png" width="600px">

A perceptron in its simplest implementation computes a weighted sum. The purpose of the activation function is to determine if the weighted sum is greater than some threshold. If it is then we output `True` (1) otherwise we output `False` (0).

- In other words the output is binary and the activation function resembles a **step function**.

---

### Mathematical Steps

1) Compute weighted sum

$$z = X^TW$$

$$z = w_1x_1 + w_2x_2 + \ldots + w_nx_n$$

2) Define threshold activation function

$$
h(z) =
\begin{cases} 
0 & \text{if } z < \text{Threshold} \\
1 & \text{if } z \geq \text{Threshold}
\end{cases}
$$

3) Update weights as defined by gradient descent

$$
w_{i, j} = w_{i, j} - \alpha (\hat{y}_j - y_j)x_i
$$

---

While this perceptron model we have created works beautifully in linearly separable problem sets, it struggles heavily in situations where datapoints are not linearly separable.

To show this, lets create some code!


```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification

# Generate a linearly separable dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, n_redundant=0, n_clusters_per_class=1, random_state=42)

# Train a perceptron model
model = Perceptron()
model.fit(X, y)

# Extract coefficients and intercept
w = model.coef_[0]
b = model.intercept_[0]

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_values = np.linspace(x_min, x_max, 100)
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, 'r-', label='Decision Boundary')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron Decision Boundary")
plt.legend()
plt.show()

```


    
![png](/notebooks/media/lesson_4_3_0.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_moons

# Generate a non-linearly separable dataset
X, y = make_moons(n_samples=100, noise=0.1, random_state=40)

# Train a perceptron model
model = Perceptron()
model.fit(X, y)

# Extract coefficients and intercept
w = model.coef_[0]
b = model.intercept_[0]

# Print slope and intercept
slope = -w[0] / w[1]
intercept = -b / w[1]
print(f"Slope: {slope}")
print(f"Intercept: {intercept}")

# Plot the dataset
plt.figure(figsize=(8, 6))
plt.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# Plot decision boundary
x_min, x_max = X[:, 0].min() - 1, X[:, 0].max() + 1
y_min, y_max = X[:, 1].min() - 1, X[:, 1].max() + 1
x_values = np.linspace(x_min, x_max, 100)
y_values = -(w[0] * x_values + b) / w[1]
plt.plot(x_values, y_values, 'r-', label='Decision Boundary')

plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.title("Perceptron on Non-Linearly Separable Data")
plt.legend()
plt.show()

```

    Slope: 0.37930585659554084
    Intercept: 0.542378029996429



    
![png](/notebooks/media/lesson_4_4_1.png)
    



```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_classification

# Generate a dataset
X, y = make_classification(n_samples=100, n_features=2, n_classes=2, 
                           n_redundant=0, n_clusters_per_class=1, random_state=20)

# Initialize the perceptron model with partial_fit capability
model = Perceptron(max_iter=1, tol=None)

# Pre-compute all model states for animation
n_iterations = 20
model_states = []

# Store initial state before training
w = np.zeros(2)  # Initial zero weights
b = 0            # Initial zero bias
model_states.append((w.copy(), b))

# Train model one iteration at a time and save states
for i in range(n_iterations):
    model.partial_fit(X, y, classes=np.unique(y))
    model_states.append((model.coef_[0].copy(), model.intercept_[0]))

# Setup figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Perceptron Learning Evolution")

# Scatter plot of data points
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# Initialize decision boundary line
x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
boundary, = ax.plot([], [], 'r-', label='Decision Boundary')
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

# Animation function that uses pre-computed states
def update(i):
    w, b = model_states[i]
    
    # Handle potential division by zero
    if abs(w[1]) < 1e-10:  # If w[1] is very close to zero
        # Draw a vertical line if needed
        if w[0] != 0:
            x_const = -b / w[0]
            boundary.set_data([x_const, x_const], [X[:, 1].min() - 1, X[:, 1].max() + 1])
        else:
            # If both w[0] and w[1] are zero, no line to draw
            boundary.set_data([], [])
    else:
        # Normal case - calculate slope and intercept
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        y_values = slope * x_values + intercept
        boundary.set_data(x_values, y_values)
    
    # Update iteration text
    iteration_text.set_text(f'Iteration: {i}')
    
    return boundary, iteration_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_iterations+1, interval=500, blit=True)

# Save the animation (this will ensure it's actually rendered)
try:
    ani.save('./media/perceptron_training_on_separable.gif', writer='pillow', fps=2)
    print("Animation saved as 'perceptron_training_on_separable.gif' in media folder.")
except Exception as e:
    print(f"Could not save animation: {e}")
    print("Try installing pillow: pip install pillow")

# This will display in interactive environments
plt.tight_layout()
plt.show()
```

    Animation saved as 'perceptron_training_on_separable.gif' in media folder.



    
![png](/notebooks/media/lesson_4_5_1.png)
    


The gif below shows the evolution of the decision boundary generated by the perceptron over time. Each iteration represents one pass through the entire dataset.

![file](/notebooks/media/perceptron_training_on_separable.gif)

In the visualization, the decision boundary line represents exactly where this weighted sum equals zero (z = 0). Points on one side of the line have a weighted sum greater than zero and are classified as one class, while points on the other side have a weighted sum less than zero and are classified as the other class.

The animation shows how the perceptron algorithm adjusts the weights and bias with each iteration, moving the decision boundary to better separate the two classes. This captures the essence of the perceptron learning rule: when the model makes a mistake, it updates its weights proportionally to correct the error.

**However note that this only works reasonably well with linearly separable data!**

Beneath is the same experiment, but run on non-linearly separable circlularly graphed datapoints.


```python
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from sklearn.linear_model import Perceptron
from sklearn.datasets import make_circles

# Generate a dataset
X, y = make_circles(n_samples=500, noise=0.1)
# Initialize the perceptron model with partial_fit capability
model = Perceptron(max_iter=1, tol=None)

# Pre-compute all model states for animation
n_iterations = 20
model_states = []

# Store initial state before training
w = np.zeros(2)  # Initial zero weights
b = 0            # Initial zero bias
model_states.append((w.copy(), b))

# Train model one iteration at a time and save states
for i in range(n_iterations):
    model.partial_fit(X, y, classes=np.unique(y))
    model_states.append((model.coef_[0].copy(), model.intercept_[0]))

# Setup figure
fig, ax = plt.subplots(figsize=(8, 6))
ax.set_xlim(X[:, 0].min() - 1, X[:, 0].max() + 1)
ax.set_ylim(X[:, 1].min() - 1, X[:, 1].max() + 1)
ax.set_xlabel("Feature 1")
ax.set_ylabel("Feature 2")
ax.set_title("Perceptron Learning Evolution")

# Scatter plot of data points
scatter = ax.scatter(X[:, 0], X[:, 1], c=y, cmap=plt.cm.Paired, edgecolors='k')

# Initialize decision boundary line
x_values = np.linspace(X[:, 0].min() - 1, X[:, 0].max() + 1, 100)
boundary, = ax.plot([], [], 'r-', label='Decision Boundary')
iteration_text = ax.text(0.02, 0.95, '', transform=ax.transAxes)
ax.legend()

# Animation function that uses pre-computed states
def update(i):
    w, b = model_states[i]
    
    # Handle potential division by zero
    if abs(w[1]) < 1e-10:  # If w[1] is very close to zero
        # Draw a vertical line if needed
        if w[0] != 0:
            x_const = -b / w[0]
            boundary.set_data([x_const, x_const], [X[:, 1].min() - 1, X[:, 1].max() + 1])
        else:
            # If both w[0] and w[1] are zero, no line to draw
            boundary.set_data([], [])
    else:
        # Normal case - calculate slope and intercept
        slope = -w[0] / w[1]
        intercept = -b / w[1]
        y_values = slope * x_values + intercept
        boundary.set_data(x_values, y_values)
    
    # Update iteration text
    iteration_text.set_text(f'Iteration: {i}')
    
    return boundary, iteration_text

# Create animation
ani = animation.FuncAnimation(fig, update, frames=n_iterations+1, interval=500, blit=True)

# Save the animation (this will ensure it's actually rendered)
try:
    ani.save('./media/perceptron_training_on_nonseparable.gif', writer='pillow', fps=2)
    print("Animation saved as 'perceptron_training_on_nonseparable.gif'")
except Exception as e:
    print(f"Could not save animation: {e}")
    print("Try installing pillow: pip install pillow")

# This will display in interactive environments
plt.tight_layout()
plt.show()
```

    Animation saved as 'perceptron_training_on_nonseparable.gif'



    
![png](/notebooks/media/lesson_4_7_1.png)
    


Unlike the experiment on two separable datasets, a single line is insufficent to separate non-linearly separable date!

<img src="/notebooks/media/perceptron_training_on_nonseparable.gif">


## Pitfalls of Simple Perceptrons

Hopefully it should be apparent that a single perceptron generates a single line or (with 3d datapoints a plane) that fails to capture relationships that are not linearly separable.

Suppose for our first example, **we had 3 clusters instead of two, how many lines would be required to separate the three clusters?**

- **In the best case scenario we would need two lines to separate each cluster into its own region.**

Similarly for 4 clusters we would want three 3 lines (3 perceptrons), and the trend continues (assuming linearly separable datapoints).

However, there are many complex relationships that cannot be represented as or boiled down to linear relationships. Is there a way we could alter or enhance the logic of a basic binary classifier esque perceptron to accomodate this non-linearity?

One hypothesis is that we could simply add more layers. In other words **we could apply a linear function to the sums of multiple other linear functions. (the weighted sum of several weighted sums).**

- *Would such a function be non-linear?*

- Applying linearity to a linear function may help abstract and define more complicated linear relationships, but it is not sufficient to express non-linearity. In other words, **stacking more linear transformations (even with multiple layers) would still result in an overall linear transformation, as a composition of linear functions remains linear**.

## Introducing non-linearity

Ask yourself, what function can take in some linear inputs but transform them into non-linear outputs?

- Well the simple answer would be parabolas and other quadratic functions.

However, a simple quadratic function does not cover the capabilities or features we want from this transformative function, which are as follows:

1) The function should preferably output a probability (a number from 0 to 1).

2) The function should be relatively computationally inexpensive (no complex parabolas or super high dimensional stuff)

3) The curve of the function should be smooth and ideally differentiable along its entirety (no holes or undefined points in the function or its gradient).

4) The function should overall just be simple and predictable in how it behaves.

Fortunately we have already tackled this topic in prior lessons when discussing how non-linearity can be applied to any weighted sum.

The only difference here is the context in which the weighted sum is being computed!

Recall the following:

Sigmoid function: $f(z) = \frac{1}{1 + e^{-z}}$

Rectified Linear Unit (ReLU): $f(z) = \max(0, z)$

### Graphs of Nonlinear functions:


```python
import numpy as np
import matplotlib.pyplot as plt

# Graph of sigmoid function
x = np.arange(-10, 10, 0.01)
y = 1 / (1 + np.exp(-x))

# Ensure that plot is representative (make plot wide to show the smoothness of the curve)
plt.figure(figsize=(10,2))

plt.plot(x, y)
plt.show()

# Graph of Relu function
x = np.arange(-10, 10, 0.01)
y = np.maximum(0, x)

plt.plot(x, y)
plt.show()
```


    
![png](/notebooks/media/lesson_4_12_0.png)
    



    
![png](/notebooks/media/lesson_4_12_1.png)
    


Our conversations thus far, especially in the prior lesson have revolved around the usage of the sigmoid function!

- The loss function and gradient calculations we have discussed have been with the assumption of an underlying sigmoid activation.

However, while sigmoid has been a useful tool in our discussions so far, it's important to consider whether it's always the best choice. As we move forward, we'll examine its strengths and weaknesses—particularly in terms of vanishing gradients and computational efficiency—and compare it to ReLU, another widely used activation function that addresses some of sigmoid’s limitations.

## Expanding and Implementing a MLP Layer

While a singular perceptron can be useful to capture a simple, singular trend, we can initialize and embed several perceptrons in the same layer.

- The only difference in how the perceptrons calculate the weighted sum would be the weights in themselves.

![Multiperceptron Diagram](/notebooks/media/multi_perceptron_diagram.png)

Each weighted sum here: $z_1$ and $z_2$ would apply two distinct sets of weights to the same input!

- In doing so, each perceptron will create a different "line" or classification boundary which we will come to see can be leveraged to capture various features in the data.

A layer in which every input neuron (in this case each neuron represents a data point) is connected to every output neuron is called a **dense layer**.

- Since both $z_1$ and $z_2$ operate on all of the neurons behind them, those input neurons and these perceptrons form a dense layer.

If you read lesson 0, it also makes intuitive sense to place all our weights in the same matrix, where the number of rows is the number of weights as a whole and the number of columns is the number of perceptrons or output nodes.

Lets implement a basic dense layer!


```python
import tensorflow as tf
import torch
from torch import nn

physical_devices = tf.config.list_physical_devices('GPU')
print(physical_devices)

class DenseLayer(tf.keras.layers.Layer):
    def __init__(self, n_inputs, n_outputs):
        super(DenseLayer, self).__init__()
    
        # Initialize weights
        # (The add_weight method is provided in the Layer parent class)
        self.weights = self.add_weight([n_inputs, n_outputs])
        self.bias = self.add_weight([1, n_outputs])
        
    def call(self, inputs):
        
        # Pass the input data into the weights and get that weighted sum
        z = tf.matmul(inputs, self.weights) + self.bias
        
        # Return the sigmoid(z) (i.e. apply the non-linearity)
        return tf.math.sigmoid(z)
    
class DenseLayer(nn.Module):
    def __init__(self, n_inputs, n_outputs):
        super(DenseLayer, self).__init__()
        
        # Initialize the weights and biases
        # randn creates an explicitly random set of values in matrix form
        self.weights = nn.Parameter(torch.randn(n_inputs, n_outputs, requires_grad=True))
        self.bias = nn.Parameter(torch.randn(1, n_outputs, requires_grad=True))
        
    def forward(self, inputs):
        
        # Pass the input data into the weights and get that weighted sum
        z = tf.matmul(inputs, self.weights) + self.bias
        
        # Return the sigmoid(z) (i.e. apply the non-linearity)
        return tf.math.sigmoid(z)
```

    [PhysicalDevice(name='/physical_device:GPU:0', device_type='GPU')]


Thankfully both tensorflow and pytorch have built in equivalents of a simple linear, dense layer so we don't have to write this code ourselves:

Tensorflow:

```python

    # Creating a dense layer with two output perceptrons
    layer = tf.keras.layers.Dense(units=2)

```

Pytorch:

```python

    # Creating a dense layer which accepts in 10 values and evaluates them on two perceptrons.
    layer = nn.Linear(in_features = 10, out_features= 2)

```

Ultimately a deep neural network is a stack of several layers of these perceptrons and the accompanying non-linearity functions to transform the initial input values in more weird ways.

The way we can combine the effects of these layers sequentially could be tedious, so both tensorflow and pytorch have built in `Sequential` abstractions that properly chain these layers.

Tensorflow

```python

import tensorflow as tf

# Create a Sequential model in TensorFlow with two layers
model = tf.keras.Sequential([
    # First Dense layer: Takes in an unspecified number of inputs, outputs to n units
    tf.keras.layers.Dense(n),
    
    # Second Dense layer: Takes the output of the previous layer and maps it to 2 output units
    tf.keras.layers.Dense(2)
])


```

Pytorch

```python

import torch.nn as nn

# Create a Sequential model in PyTorch
model = nn.Sequential(
    # First Linear layer: Takes in m inputs, outputs to n perceptrons (units)
    nn.Linear(m, n),  # The number of input features is m, the output is n units
    
    # ReLU Activation: Applies the ReLU non-linearity element-wise to the outputs of the previous layer
    nn.ReLU(),
    
    # Second Linear layer: Takes the n units from the previous layer and generates 2 output units
    nn.Linear(n, 2)   # The input here is n units (from the previous layer), and output is 2 units
)


```

Note that the code here generates a model, not the data itself. The data must be appropriately generated / retrieved and fed to a model with the appropriate number of inputs. 

- The Tensorflow will accomodate itself to any number of inputs.

- The Pytorch model shown prior will require $m$ inputs specifically.

**For more complex tasks you probably want a model with more depth! Why? Simply because the application of more layers of non-linearity helps the model fine-tune its task of classification.**


```python
import torch
import torch.nn as nn
import torch.nn.functional as F
import matplotlib.pyplot as plt
from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

# Generate non-linearly separable data (moons)
X, y = make_moons(n_samples=500, noise=0.2, random_state=42)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize features
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Convert to torch tensors
X_train_tensor = torch.tensor(X_train_scaled, dtype=torch.float32)
y_train_tensor = torch.tensor(y_train, dtype=torch.long)
X_test_tensor = torch.tensor(X_test_scaled, dtype=torch.float32)
y_test_tensor = torch.tensor(y_test, dtype=torch.long)

# Define a simple MLP model
model = nn.Sequential(
    nn.Linear(2, 10),
    nn.ReLU(), # Try swapping this between nn.Sigmoid(), nn.Tanh(), or nn.ReLU()
    nn.Linear(10, 2)
)

# Define loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

# Train the model
epochs = 100
for epoch in range(epochs):
    optimizer.zero_grad()
    output = model(X_train_tensor)
    loss = criterion(output, y_train_tensor)
    loss.backward()
    optimizer.step()

# Create a meshgrid to visualize decision boundary
import numpy as np

h = 0.01
x_min, x_max = X_train_scaled[:, 0].min() - 1, X_train_scaled[:, 0].max() + 1
y_min, y_max = X_train_scaled[:, 1].min() - 1, X_train_scaled[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))
grid_tensor = torch.tensor(np.c_[xx.ravel(), yy.ravel()], dtype=torch.float32)
Z = model(grid_tensor).detach().numpy()
Z = np.argmax(Z, axis=1)
Z = Z.reshape(xx.shape)

# Plot decision boundary and training data
plt.figure(figsize=(8, 6))
plt.contourf(xx, yy, Z, alpha=0.4, cmap=plt.cm.Paired)
plt.scatter(X_train_scaled[:, 0], X_train_scaled[:, 1], c=y_train, edgecolors='k', cmap=plt.cm.Paired)
plt.title("MLP Decision Boundary on Non-Linear Data")
plt.xlabel("Feature 1")
plt.ylabel("Feature 2")
plt.show()

```


    
![png](/notebooks/media/lesson_4_17_0.png)
    


## Backpropagation

In our initial discussion of how gradient descent is calculated for logistic regression we set out with the following roadmap:

---

"""Our final goals are to calculate the following:

- The derivative of the error function with respect to $W$: $\frac{\partial Error}{\partial W}$
- The derivative of the error function with respect to $b$: $\frac{\partial Error}{\partial b}$

Based on the chain rule our roadmap looks like:

$$\frac{\partial Error}{\partial W} = \sum_{i = 1}^n\frac{\partial Error}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_i} \frac{\partial z_i}{\partial W}$$

$$\frac{\partial Error}{\partial b} = \sum_{i = 1}^n\frac{\partial Error}{\partial \hat{y}_i} \frac{\partial \hat{y}_i}{\partial z_i} \frac{\partial z_i}{\partial b}$$

**Finding the derivative of the error function with respect to the weights implies I need to find the derivative of the error with respect to $y_i$ ($\frac{\partial Error}{\partial y_i}$), the derivative of $y_i$ with respect to $z_i$ ($\frac{\partial \hat{y}_i}{\partial z_i}$) and the derivative of $z_i$ with respect to each weight ($\frac{\partial z_i}{\partial W}$)**

**Finding the derivative of the error function with respect to the weights implies the same steps except I need to find the derivative of $z_i$ with respect to the bias (trivial enough :smile:).**"""

---

In that case the activation function of logistic regression acted as a sort of final layer enacted on the weighted sum computed in a perceptron.

But we could apply this methodology of expanding via chain rule to a sequence of computations and activations!

In a neural network with multiple layers, we will apply the chain rule for each layer, starting from the output and moving backward to the input layer. For each layer, you compute the gradient of the error with respect to the weights and biases. The process is broken down as follows:

1. Forward Pass: Compute the activations for all layers my sending data "forwards" from left to right in the network.

2. Backward Pass: For each layer starting from the output, recursively compute the gradient of the error by applying chain rule.

3. Backpropagation: 

- Suppose three functions apply to some data input, $x$,  in the form $h(g(f(x)))$

- $f$ represents the first transformation, i.e. the first layer of our network.

- $g$ represents the hidden transformation i.e. the hidden layer of our network.

- $h$ represents the outermost layer i.e. the final transformation of our network.

If we wanted to calculate $\frac{\partial h}{\partial x}$ we would do so by applying chain rule from $h$ all the way down to $x$

<img src="/notebooks/media/backpropagation_explanation_1.png" width="400px">

- **Backpropagation is just a fancy application of chain rule for the purposes of evaluating error from each layer in the network.**

- **Instead of x, imagine the actual data point, and instead of simply $f$, $g$, and $h$ imagine the error functions at each layer!**

Think of the forward pass as a series of transformations applied to your input data, producing the final output. The backpropagation process, on the other hand, is like you trying to trace how much each transformation (layer) contributed to the error in the output, and then adjusting the parameters (weights and biases) of those transformations to minimize that error.

The key idea is that the error at the output layer is caused by all of the previous layers, and to fix that error, you need to adjust the parameters at each layer. **Since each layer is a function of its previous layer (or input data), the chain rule allows you to compute how to adjust each weight and bias.**

- With more layers the depth of this chain rule computation increases significantly, but ultimately the computations of calculating the error for several nodes in the same layer can often be done in parallel via GPUs.

- There are more advanced ways to exploit the parallelizability of neural models as well!

## The ML Learning Pipeline

As discussed in prior lessons, the mathematical foundation of the learning process for models attempting to learn from a set of data is rigorously tested and technically very complex.

However, we could summarize our learnings in a few points:

- Models apply mathematical functions (i.e. weighted sums, non-linearity functions, e.t.c)

- We can measure how well a particular function (visualize it as a line in 2d space) fitted to the given data.

- As we get fed more data the function's slope or weights can be corrected to match the data to the best of our ability.

- Instead of trying to correct the function itself, we can instead map the loss of the function (i.e. the difference between each prediction and our expectations). This loss turns out to be a function itself.

- If we can minimize loss, we are inversely maximizing our correctness (which can be measured in many ways but can be generally be thought of as "best fit")

- We use gradient descent, and algorithmic technique, to find values that minimize the loss function with any given data point.

- Gradient descent operates to change the weights themselves (i.e. think of $y = mx + b$, we change the slope by modifying m, not x).

- Gradient descent works by going over the data in iterations of batches (it could update based on each datapoint, or once after each epoch over the entire dataset or choose a batch size somewhere in between).

- Advanced techniques help correct the pitfalls of classical gradient descent (which arise from problematic functions or erratic step size changes).

When you put all this together here is the process we are aiming to achieve with our code:

1) **Data preparation** (loading and preprocessing the data),

2) **Model definition / initialization** (creating the architecture like the ones in TensorFlow/PyTorch),

3) **Loss function and optimizer definition / initialization** (setting up the loss and optimization algorithm),

4) **Training loop** (iterating through epochs, updating weights, and calculating loss),

5) **Evaluation** (measuring accuracy on the validation set).

The rest of this lesson will cover examples of training a model on the **MNIST** dataset. The information regarding this dataset can be found
on these sites:

- [Wikipedia](https://en.wikipedia.org/wiki/MNIST_database)
- [Kaggle](https://www.kaggle.com/datasets/hojjatk/mnist-dataset)
- [Tensorflow](https://www.tensorflow.org/datasets/catalog/mnist)

We will start by attempting to build a network from scratch to the best of our extent, after understanding the logic behind the raw and tediously coded version of the program we can make various optimizations within the code to accomplish the task much, much quicker (in terms of development time).

## MNIST Model from (nearly) Scratch

### Data Preparation and Analysis


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms
import matplotlib.pyplot as plt
from PIL import Image

"""
Before doing anything else, we need to load in the MNIST data
and apply some basic transformations.
"""

# Create the transformation object (which is more like a function we will apply)
# to the image set.
transform = transforms.Compose([
    transforms.ToTensor(), # This turns each value in the applied upon array into a tensor.
    transforms.Normalize((0.5,), (0.5,)) # This normalizes pixel values from 0 to 255 to be from 0 to 1.
])

# Load MNIST data and epply the aforementioned transforms!
train_dataset = datasets.MNIST(root="./MNIST_data", train=True, download=True, transform=transform)
test_dataset = datasets.MNIST(root="./MNIST_data", train=False, download=True, transform=transform)

# Use a set of dataloader objects that act as a sort of iterator to control the flow of data (adjust the batch size)
# (This will be useful later)
train_loader = DataLoader(train_dataset, batch_size=64, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=64, shuffle=False)

# Check the structure of the training dataset
print(f"Training dataset type: {type(train_dataset)}")
print(f"Number of training samples: {len(train_dataset)}")

# Check the first data sample (image and label)
image, label = train_dataset[0]
print("Example in training dataset:\n", train_dataset[0])

"""

The statement above should give some insight on the form of the data!
We can see that a single image is defined by a 2D array of singular values within an array, which is
itself a tensor AND by a singular second value from 0 to 9 (that represents the true label of the data).

The fact that each image has inner most pixels represented by single values should be indicative
that the pixels are single channel (as opposed to three channels: r, g, b or 4 channels: r, g, b, a)

- Instead each image is 28 by 28 pixels, single channel (monochrome)

"""

print(f"Image shape: {image.shape}")  # Shape of the first image
print(f"Label: {label}")  # Label for the first image

"""
We can render the image sample using something like pillow (PIL) and matplotlib
"""

# Let's get a sample from the train dataset
image, label = train_dataset[0]  # Access the first image and label

# Convert the tensor to a PIL image (remove the normalization by scaling back)
print("Dimensions of image pre-squeeze: ", image.shape)
image = image.squeeze()  # Remove the channel dimension 
# (i.e. the third pair of brackets in the traditional tensor which would be useful for multiple channels but is un-necessary for monochrome), 
print("Dimensions of image post-squeeze: ", image.shape)
# now it's (28, 28) instead of (1, 28, 28) since the channel dimension is removed.
print(image)

# Pillow expects images in range [0, 255] for uint8 format, so we scale the tensor to that range
image = transforms.ToPILImage()(image)  # Converts tensor to a PIL Image

# Optionally, you can also visualize it using matplotlib
plt.imshow(image, cmap='gray')
plt.title(f"Label: {label}")
plt.axis('off')  # Hide axes
plt.show()
```


    ---------------------------------------------------------------------------

    RuntimeError                              Traceback (most recent call last)

    Cell In[1], line 5
          3 import torch.optim as optim
          4 from torch.utils.data import DataLoader
    ----> 5 from torchvision import datasets, transforms
          6 import matplotlib.pyplot as plt
          7 from PIL import Image


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/__init__.py:10
          7 # Don't re-order these, we need to load the _C extension (done when importing
          8 # .extensions) before entering _meta_registrations.
          9 from .extension import _HAS_OPS  # usort:skip
    ---> 10 from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
         12 try:
         13     from .version import __version__  # noqa: F401


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/_meta_registrations.py:163
        153     torch._check(
        154         grad.dtype == rois.dtype,
        155         lambda: (
       (...)    158         ),
        159     )
        160     return grad.new_empty((batch_size, channels, height, width))
    --> 163 @torch.library.register_fake("torchvision::nms")
        164 def meta_nms(dets, scores, iou_threshold):
        165     torch._check(dets.dim() == 2, lambda: f"boxes should be a 2d tensor, got {dets.dim()}D")
        166     torch._check(dets.size(1) == 4, lambda: f"boxes should have 4 elements in dimension 1, got {dets.size(1)}")


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/library.py:828, in register_fake.<locals>.register(func)
        826 else:
        827     use_lib = lib
    --> 828 use_lib._register_fake(op_name, func, _stacklevel=stacklevel + 1)
        829 return func


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/library.py:198, in Library._register_fake(self, op_name, fn, _stacklevel)
        195 else:
        196     func_to_register = fn
    --> 198 handle = entry.fake_impl.register(func_to_register, source)
        199 self._registration_handles.append(handle)


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/_library/fake_impl.py:31, in FakeImplHolder.register(self, func, source)
         25 if self.kernel is not None:
         26     raise RuntimeError(
         27         f"register_fake(...): the operator {self.qualname} "
         28         f"already has an fake impl registered at "
         29         f"{self.kernel.source}."
         30     )
    ---> 31 if torch._C._dispatch_has_kernel_for_dispatch_key(self.qualname, "Meta"):
         32     raise RuntimeError(
         33         f"register_fake(...): the operator {self.qualname} "
         34         f"already has an DispatchKey::Meta implementation via a "
       (...)     37         f"register_fake."
         38     )
         40 if torch._C._dispatch_has_kernel_for_dispatch_key(
         41     self.qualname, "CompositeImplicitAutograd"
         42 ):


    RuntimeError: operator torchvision::nms does not exist


### Defining the Linear Neural Network


```python
from torch import nn

class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        
        # Input to hidden layer
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()  # Activation function
        
        # Hidden to output layer
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        # Pass through the network
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

```

### Initializing components of training (model, optimizer, regularization, e.t.c.)


```python
import time

"""

Our model accepts inputs in the form of a singular dimension array.

This means we have to flatten our 28 by 28 image into a single dimensional array equivalent which will have a length of 784.

Our single hidden layer can any number of nodes

Our output layer should have 10 nodes, each of which corresponds to a number (0-9)

"""

# Configure and initialize the model
input_size, hidden_size, output_size = 28*28, 100, 10
model = SimpleNN(input_size, hidden_size, output_size)


# Initialize the loss function (often also called "criterion")
loss_function = nn.CrossEntropyLoss()

# For now lets stick with no optimization (i.e. SGD or ADAM) and define the hard coded learning rate
lr = 0.01

# With our hyperparameters (in this case just the learning rate) set up, lets loop!
n_epochs = 10

# Lets time our training process for fun
start = time.time()

for epoch in range(n_epochs):
    # We go through the entire dataset by iterating several times
    # over the loader. Think of the loader as a valve that moderates the
    # batch size as we initialized it from before (64 samples per iteration).
    for images, labels in train_loader:
        
        # Perform a forward pass by getting outputted values and then computing loss (the hard part)
        # Flatten images to 1D (28*28)
        images = images.view(-1, 28*28) # (First flatten the image)
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # Then backpropagate (assign the loss)
        loss.backward()
    
        # Update the weights manually
        with torch.no_grad(): # this disables gradient tracking (we only care about current grad) and changing it
            for param in model.parameters(): # go through each parameter (neuron)
                param -= lr * param.grad # GRADIENT DESCENT BABY!
        
        
        # Clear up the gradients (i.e. reset them) for the next batch coming in
        model.zero_grad()
        
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {loss.item():.4f}")

print(f"Time taken for training: {time.time() - start}")

```

    Epoch [1/10], Loss: 0.5120
    Epoch [2/10], Loss: 0.1511



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[18], line 36
         30 start = time.time()
         32 for epoch in range(n_epochs):
         33     # We go through the entire dataset by iterating several times
         34     # over the loader. Think of the loader as a valve that moderates the
         35     # batch size as we initialized it from before (64 samples per iteration).
    ---> 36     for images, labels in train_loader:
         37 
         38         # Perform a forward pass by getting outputted values and then computing loss (the hard part)
         39         # Flatten images to 1D (28*28)
         40         images = images.view(-1, 28*28) # (First flatten the image)
         41         outputs = model(images)


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/utils/data/dataloader.py:734, in __next__(self)
        728     def __len__(self) -> int:
        729         return len(self._index_sampler)
        731     def __getstate__(self):
        732         # TODO: add limited pickling support for sharing an iterator
        733         # across multiple threads for HOGWILD.
    --> 734         # Probably the best way to do this is by moving the sample pushing
        735         # to a separate thread and then just sharing the data queue
        736         # but signalling the end is tricky without a non-blocking API
        737         raise NotImplementedError("{} cannot be pickled", self.__class__.__name__)
        740 class _SingleProcessDataLoaderIter(_BaseDataLoaderIter):


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/utils/data/dataloader.py:790, in _next_data(self)
        770 class _MultiProcessingDataLoaderIter(_BaseDataLoaderIter):
        771     r"""Iterates once over the DataLoader's dataset, as specified by the sampler."""
        773     # NOTE [ Data Loader Multiprocessing Shutdown Logic ]
        774     #
        775     # Preliminary:
        776     #
        777     # Our data model looks like this (queues are indicated with curly brackets):
        778     #
        779     #                main process                              ||
        780     #                     |                                    ||
        781     #               {index_queue}                              ||
        782     #                     |                                    ||
        783     #              worker processes                            ||     DATA
        784     #                     |                                    ||
        785     #            {worker_result_queue}                         ||     FLOW
        786     #                     |                                    ||
        787     #      pin_memory_thread of main process                   ||   DIRECTION
        788     #                     |                                    ||
        789     #               {data_queue}                               ||
    --> 790     #                     |                                    ||
        791     #                data output                               \/
        792     #
        793     # P.S. `worker_result_queue` and `pin_memory_thread` part may be omitted if
        794     #      `pin_memory=False`.
        795     #
        796     #
        797     # Terminating multiprocessing logic requires very careful design. In
        798     # particular, we need to make sure that
        799     #
        800     #   1. The iterator gracefully exits the workers when its last reference is
        801     #      gone or it is depleted.
        802     #
        803     #      In this case, the workers should be gracefully exited because the
        804     #      main process may still need to continue to run, and we want cleaning
        805     #      up code in the workers to be executed (e.g., releasing GPU memory).
        806     #      Naturally, we implement the shutdown logic in `__del__` of
        807     #      DataLoaderIterator.
        808     #
        809     #      We delay the discussion on the logic in this case until later.
        810     #
        811     #   2. The iterator exits the workers when the loader process and/or worker
        812     #      processes exits normally or with error.
        813     #
        814     #      We set all workers and `pin_memory_thread` to have `daemon=True`.
        815     #
        816     #      You may ask, why can't we make the workers non-daemonic, and
        817     #      gracefully exit using the same logic as we have in `__del__` when the
        818     #      iterator gets deleted (see 1 above)?
        819     #
        820     #      First of all, `__del__` is **not** guaranteed to be called when
        821     #      interpreter exits. Even if it is called, by the time it executes,
        822     #      many Python core library resources may already be freed, and even
        823     #      simple things like acquiring an internal lock of a queue may hang.
        824     #      Therefore, in this case, we actually need to prevent `__del__` from
        825     #      being executed, and rely on the automatic termination of daemonic
        826     #      children.
        827     #
        828     #      Thus, we register an `atexit` hook that sets a global flag
        829     #      `_utils.python_exit_status`. Since `atexit` hooks are executed in the
        830     #      reverse order of registration, we are guaranteed that this flag is
        831     #      set before library resources we use are freed (which, at least in
        832     #      CPython, is done via an `atexit` handler defined in
        833     #      `multiprocessing/util.py`
        834     #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/util.py#L320-L362
        835     #      registered when an object requiring this mechanism is first
        836     #      created, e.g., `mp.Queue`
        837     #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/context.py#L100-L103
        838     #      https://github.com/python/cpython/blob/c606624af8d4cb3b4a052fb263bb983b3f87585b/Lib/multiprocessing/queues.py#L29
        839     #      )
        840     #
        841     #      So in `__del__`, we check if `_utils.python_exit_status` is set or
        842     #      `None` (freed), and perform no-op if so.
        843     #
        844     #      However, simply letting library clean-up codes run can also be bad,
        845     #      because such codes (i.e., `multiprocessing.util._exit_function()`)
        846     #      include join putting threads for `mp.Queue`, which can be blocking.
        847     #      Hence, the main process putting threads are called with
        848     #      `cancel_join_thread` at creation.  See later section
        849     #      [ 3b. A process won't hang when putting into a queue; ]
        850     #      for more details.
        851     #
        852     #      Here are two example cases where library clean-up codes can run
        853     #      before `__del__` is called:
        854     #
        855     #        1. If we hold onto a reference to the iterator, it more often
        856     #           than not tries to do `multiprocessing` library cleaning before
        857     #           clearing the alive referenced objects (https://github.com/pytorch/pytorch/issues/48666)
        858     #           and thus prevents our cleaning-up code to run first.
        859     #
        860     #        2. A similar issue araises when a `DataLoader` is used in a subprocess.
        861     #           When a process ends, it shuts the all its daemonic children
        862     #           down with a SIGTERM (instead of joining them without a timeout).
        863     #           Simiarly for threads, but by a different mechanism. This fact,
        864     #           together with a few implementation details of multiprocessing, forces
        865     #           us to make workers daemonic. All of our problems arise when a
        866     #           DataLoader is used in a subprocess, and are caused by multiprocessing
        867     #           code which looks more or less like this:
        868     #
        869     #               try:
        870     #                   your_function_using_a_dataloader()
        871     #               finally:
        872     #                   multiprocessing.util._exit_function()
        873     #
        874     #           The joining/termination mentioned above happens inside
        875     #           `_exit_function()`. Now, if `your_function_using_a_dataloader()`
        876     #           throws, the stack trace stored in the exception will prevent the
        877     #           frame which uses `DataLoaderIter` to be freed. If the frame has any
        878     #           reference to the `DataLoaderIter` (e.g., in a method of the iter),
        879     #           its  `__del__`, which starts the shutdown procedure, will not be
        880     #           called. That, in turn, means that workers aren't notified. Attempting
        881     #           to join in `_exit_function` will then result in a hang.
        882     #
        883     #           For context, `_exit_function` is also registered as an `atexit` call.
        884     #           So it is unclear to me (@ssnl) why this is needed in a finally block.
        885     #           The code dates back to 2008 and there is no comment on the original
        886     #           PEP 371 or patch https://bugs.python.org/issue3050 (containing both
        887     #           the finally block and the `atexit` registration) that explains this.
        888     #
        889     #
        890     #      Finally, another choice is to just shutdown workers with logic in 1
        891     #      above whenever we see an error in `next`. This isn't ideal because
        892     #        a. It prevents users from using try-catch to resume data loading.
        893     #        b. It doesn't prevent hanging if users have references to the
        894     #           iterator.
        895     #
        896     #   3. All processes exit if any of them die unexpectedly by fatal signals.
        897     #
        898     #      As shown above, the workers are set as daemonic children of the main
        899     #      process. However, automatic cleaning-up of such child processes only
        900     #      happens if the parent process exits gracefully (e.g., not via fatal
        901     #      signals like SIGKILL). So we must ensure that each process will exit
        902     #      even the process that should send/receive data to/from it were
        903     #      killed, i.e.,
        904     #
        905     #        a. A process won't hang when getting from a queue.
        906     #
        907     #           Even with carefully designed data dependencies (i.e., a `put()`
        908     #           always corresponding to a `get()`), hanging on `get()` can still
        909     #           happen when data in queue is corrupted (e.g., due to
        910     #           `cancel_join_thread` or unexpected exit).
        911     #
        912     #           For child exit, we set a timeout whenever we try to get data
        913     #           from `data_queue`, and check the workers' status on each timeout
        914     #           and error.
        915     #           See `_DataLoaderiter._get_batch()` and
        916     #           `_DataLoaderiter._try_get_data()` for details.
        917     #
        918     #           Additionally, for child exit on non-Windows platforms, we also
        919     #           register a SIGCHLD handler (which is supported on Windows) on
        920     #           the main process, which checks if any of the workers fail in the
        921     #           (Python) handler. This is more efficient and faster in detecting
        922     #           worker failures, compared to only using the above mechanism.
        923     #           See `DataLoader.cpp` and `_utils/signal_handling.py` for details.
        924     #
        925     #           For `.get()` calls where the sender(s) is not the workers, we
        926     #           guard them with timeouts, and check the status of the sender
        927     #           when timeout happens:
        928     #             + in the workers, the `_utils.worker.ManagerWatchdog` class
        929     #               checks the status of the main process.
        930     #             + if `pin_memory=True`, when getting from `pin_memory_thread`,
        931     #               check `pin_memory_thread` status periodically until `.get()`
        932     #               returns or see that `pin_memory_thread` died.
        933     #
        934     #        b. A process won't hang when putting into a queue;
        935     #
        936     #           We use `mp.Queue` which has a separate background thread to put
        937     #           objects from an unbounded buffer array. The background thread is
        938     #           daemonic and usually automatically joined when the process
        939     #           *exits*.
        940     #
        941     #           In case that the receiver has ended abruptly while
        942     #           reading from the pipe, the join will hang forever.  The usual
        943     #           solution for this in Python is calling  `q.cancel_join_thread`,
        944     #           which prevents automatically joining it when finalizing
        945     #           (exiting).
        946     #
        947     #           Nonetheless, `cancel_join_thread` must only be called when the
        948     #           queue is **not** going to be read from or write into by another
        949     #           process, because it may hold onto a lock or leave corrupted data
        950     #           in the queue, leading other readers/writers to hang.
        951     #
        952     #           Hence,
        953     #             + For worker processes, we only do so (for their output
        954     #               queues, i.e., `worker_result_queue`) before exiting.
        955     #             + For `pin_memory_thread`, its output queue `data_queue` is a
        956     #               `queue.Queue` that does blocking `put` if the queue is full.
        957     #               So there is no above problem, but as a result, in
        958     #               `_pin_memory_loop`, we do need to  wrap the `put` in a loop
        959     #               that breaks not only upon success, but also when the main
        960     #               process stops reading, i.e., is shutting down.
        961     #             + For loader process, we `cancel_join_thread()` for all
        962     #               `_index_queues` because the whole purpose of workers and
        963     #               `pin_memory_thread` is to serve the loader process.  If
        964     #               loader process is already exiting, we don't really care if
        965     #               the queues are corrupted.
        966     #
        967     #
        968     # Now let's get back to 1:
        969     #   how we gracefully exit the workers when the last reference to the
        970     #   iterator is gone.
        971     #
        972     # To achieve this, we implement the following logic along with the design
        973     # choices mentioned above:
        974     #
        975     # `workers_done_event`:
        976     #   A `multiprocessing.Event` shared among the main process and all worker
        977     #   processes. This is used to signal the workers that the iterator is
        978     #   shutting down. After it is set, they will not send processed data to
        979     #   queues anymore, and only wait for the final `None` before exiting.
        980     #   `done_event` isn't strictly needed. I.e., we can just check for `None`
        981     #   from the input queue, but it allows us to skip wasting resources
        982     #   processing data if we are already shutting down.
        983     #
        984     # `pin_memory_thread_done_event`:
        985     #   A `threading.Event` for a similar purpose to that of
        986     #   `workers_done_event`, but is for the `pin_memory_thread`. The reason
        987     #   that separate events are needed is that `pin_memory_thread` reads from
        988     #   the output queue of the workers. But the workers, upon seeing that
        989     #   `workers_done_event` is set, only wants to see the final `None`, and is
        990     #   not required to flush all data in the output queue (e.g., it may call
        991     #   `cancel_join_thread` on that queue if its `IterableDataset` iterator
        992     #   happens to exhaust coincidentally, which is out of the control of the
        993     #   main process). Thus, since we will exit `pin_memory_thread` before the
        994     #   workers (see below), two separete events are used.
        995     #
        996     # NOTE: In short, the protocol is that the main process will set these
        997     #       `done_event`s and then the corresponding processes/threads a `None`,
        998     #       and that they may exit at any time after receiving the `None`.
        999     #
       1000     # NOTE: Using `None` as the final signal is valid, since normal data will
       1001     #       always be a 2-tuple with the 1st element being the index of the data
       1002     #       transferred (different from dataset index/key), and the 2nd being
       1003     #       either the dataset key or the data sample (depending on which part
       1004     #       of the data model the queue is at).
       1005     #
       1006     # [ worker processes ]
       1007     #   While loader process is alive:
       1008     #     Get from `index_queue`.
       1009     #       If get anything else,
       1010     #          Check `workers_done_event`.
       1011     #            If set, continue to next iteration
       1012     #                    i.e., keep getting until see the `None`, then exit.
       1013     #            Otherwise, process data:
       1014     #                If is fetching from an `IterableDataset` and the iterator
       1015     #                    is exhausted, send an `_IterableDatasetStopIteration`
       1016     #                    object to signal iteration end. The main process, upon
       1017     #                    receiving such an object, will send `None` to this
       1018     #                    worker and not use the corresponding `index_queue`
       1019     #                    anymore.
       1020     #       If timed out,
       1021     #          No matter `workers_done_event` is set (still need to see `None`)
       1022     #          or not, must continue to next iteration.
       1023     #   (outside loop)
       1024     #   If `workers_done_event` is set,  (this can be False with `IterableDataset`)
       1025     #     `data_queue.cancel_join_thread()`.  (Everything is ending here:
       1026     #                                          main process won't read from it;
       1027     #                                          other workers will also call
       1028     #                                          `cancel_join_thread`.)
       1029     #
       1030     # [ pin_memory_thread ]
       1031     #   # No need to check main thread. If this thread is alive, the main loader
       1032     #   # thread must be alive, because this thread is set as daemonic.
       1033     #   While `pin_memory_thread_done_event` is not set:
       1034     #     Get from `worker_result_queue`.
       1035     #       If timed out, continue to get in the next iteration.
       1036     #       Otherwise, process data.
       1037     #       While `pin_memory_thread_done_event` is not set:
       1038     #         Put processed data to `data_queue` (a `queue.Queue` with blocking put)
       1039     #         If timed out, continue to put in the next iteration.
       1040     #         Otherwise, break, i.e., continuing to the out loop.
       1041     #
       1042     #   NOTE: we don't check the status of the main thread because
       1043     #           1. if the process is killed by fatal signal, `pin_memory_thread`
       1044     #              ends.
       1045     #           2. in other cases, either the cleaning-up in __del__ or the
       1046     #              automatic exit of daemonic thread will take care of it.
       1047     #              This won't busy-wait either because `.get(timeout)` does not
       1048     #              busy-wait.
       1049     #
       1050     # [ main process ]
       1051     #   In the DataLoader Iter's `__del__`
       1052     #     b. Exit `pin_memory_thread`
       1053     #          i.   Set `pin_memory_thread_done_event`.
       1054     #          ii   Put `None` in `worker_result_queue`.
       1055     #          iii. Join the `pin_memory_thread`.
       1056     #          iv.  `worker_result_queue.cancel_join_thread()`.
       1057     #
       1058     #     c. Exit the workers.
       1059     #          i.   Set `workers_done_event`.
       1060     #          ii.  Put `None` in each worker's `index_queue`.
       1061     #          iii. Join the workers.
       1062     #          iv.  Call `.cancel_join_thread()` on each worker's `index_queue`.
       1063     #
       1064     #        NOTE: (c) is better placed after (b) because it may leave corrupted
       1065     #              data in `worker_result_queue`, which `pin_memory_thread`
       1066     #              reads from, in which case the `pin_memory_thread` can only
       1067     #              happen at timing out, which is slow. Nonetheless, same thing
       1068     #              happens if a worker is killed by signal at unfortunate times,
       1069     #              but in other cases, we are better off having a non-corrupted
       1070     #              `worker_result_queue` for `pin_memory_thread`.
       1071     #
       1072     #   NOTE: If `pin_memory=False`, there is no `pin_memory_thread` and (b)
       1073     #         can be omitted
       1074     #
       1075     # NB: `done_event`s isn't strictly needed. E.g., we can just check for
       1076     #     `None` from `index_queue`, but it allows us to skip wasting resources
       1077     #     processing indices already in `index_queue` if we are already shutting
       1078     #     down.
       1080     def __init__(self, loader):
       1081         super().__init__(loader)


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/utils/data/_utils/fetch.py:52, in _MapDatasetFetcher.fetch(self, possibly_batched_index)
         50         data = self.dataset.__getitems__(possibly_batched_index)
         51     else:
    ---> 52         data = [self.dataset[idx] for idx in possibly_batched_index]
         53 else:
         54     data = self.dataset[possibly_batched_index]


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/datasets/mnist.py:146, in MNIST.__getitem__(self, index)
        143 img = _Image_fromarray(img.numpy(), mode="L")
        145 if self.transform is not None:
    --> 146     img = self.transform(img)
        148 if self.target_transform is not None:
        149     target = self.target_transform(target)


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/transforms/transforms.py:95, in Compose.__call__(self, img)
         93 def __call__(self, img):
         94     for t in self.transforms:
    ---> 95         img = t(img)
         96     return img


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/nn/modules/module.py:1773, in _wrapped_call_impl(self, *args, **kwargs)
       1771 if hook_id in self._forward_pre_hooks_with_kwargs:
       1772     args_kwargs_result = hook(self, args, kwargs)  # type: ignore[misc]
    -> 1773     if args_kwargs_result is not None:
       1774         if isinstance(args_kwargs_result, tuple) and len(args_kwargs_result) == 2:
       1775             args, kwargs = args_kwargs_result


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torch/nn/modules/module.py:1784, in _call_impl(self, *args, **kwargs)
       1782 args_result = hook(self, args)
       1783 if args_result is not None:
    -> 1784     if not isinstance(args_result, tuple):
       1785         args_result = (args_result,)
       1786     args = args_result


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/transforms/transforms.py:277, in Normalize.forward(self, tensor)
        269 def forward(self, tensor: Tensor) -> Tensor:
        270     """
        271     Args:
        272         tensor (Tensor): Tensor image to be normalized.
       (...)    275         Tensor: Normalized Tensor image.
        276     """
    --> 277     return F.normalize(tensor, self.mean, self.std, self.inplace)


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/transforms/functional.py:350, in normalize(tensor, mean, std, inplace)
        347 if not isinstance(tensor, torch.Tensor):
        348     raise TypeError(f"img should be Tensor Image. Got {type(tensor)}")
    --> 350 return F_t.normalize(tensor, mean=mean, std=std, inplace=inplace)


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/transforms/_functional_tensor.py:927, in normalize(tensor, mean, std, inplace)
        925     mean = mean.view(-1, 1, 1)
        926 if std.ndim == 1:
    --> 927     std = std.view(-1, 1, 1)
        928 return tensor.sub_(mean).div_(std)


    KeyboardInterrupt: 


### Observations from the Training Process

Training a neural network like the one above can feel slow and tedious, especially on older hardware. This is because, as we’ve discussed, neural networks involve a large number of parameters that can, and ideally *should*, be processed in parallel using GPUs. Most modern mid to high-end chips have integrated graphics that can handle light ML workloads, but I recommend investing in an affordable dedicated GPU such as an RTX 3050 because it will outperform nearly all integrated GPUs.

- If you want to leverage a GPU for ML, remember: both the model and the data must reside on the GPU. This means each time you fetch a new batch from your DataLoader, you should move it to the GPU or better yet set the GPU as your default device from the start!

- You’ve likely noticed the importance of .grad and .backward() in the training loop.

- **When updating weights, we use no_grad() because we don’t want to track gradients during the update itself.**

- After processing a batch, we’ve incorporated its contribution into the model. Therefore, we reset the gradients to ensure the next .backward() call computes fresh gradients based on the updated weights.

- Notice that we aren’t using any optimizer here to adjust the learning rate (which is considered a hyperparameter).

- Optimizers like SGD or Adam manage learning rate adjustments internally using strategies we’ve discussed.

- We can also apply regularization to the weights: L2 regularization integrates directly with most optimizers, while L1 regularization typically requires adding a manual penalty term to the loss.

- **It’s good practice to stop training early once the model reaches a satisfactory loss level — this prevents overtraining and saves time. The n_epochs parameter then becomes an upper limit rather than a fixed count of iterations.**

- While you could stop at a simple loss threshold, a more robust approach is to use early stopping with patience, which waits for sustained improvement before halting training.

- Even with just one hidden layer, having 100 hidden nodes means about $100 \times 784 \times 10 = 784{,}000$ computations per epoch for each forward pass. Reducing model complexity scales down training time proportionally.

Lets create the a significantly more optimized training process based on these optimizations (including shifting data over to the GPU if one is found)

## Optimized MNIST model training


```python
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms
from torch.utils.data import DataLoader

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

# Define the model
class SimpleNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(hidden_size, output_size)
    
    def forward(self, x):
        x = self.fc1(x)
        x = self.relu(x)
        x = self.fc2(x)
        return x

# Model hyperparameters
input_size, hidden_size, output_size = 28*28, 40, 10
learning_rate = 0.01
weight_decay = 1e-4  # L2 regularization strength
n_epochs = 50  # Max epochs (change this to a lower number if you do not have a discrete GPU)
batch_size = 64
target_loss = 0.05  # Stop if loss stays below this
patience = 3  # Number of epochs to wait before stopping

# Load dataset with transforms
transform = transforms.Compose([transforms.ToTensor()])
train_dataset = datasets.MNIST(root="./data", train=True, download=True, transform=transform)
train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=4, pin_memory=True)

# Initialize model and move to GPU if available
model = SimpleNN(input_size, hidden_size, output_size).to(device)

# Loss function
loss_function = nn.CrossEntropyLoss()

# Choose optimizer (Adam or SGD)
optimizer = optim.Adam(model.parameters(), lr=learning_rate, weight_decay=weight_decay)
# optimizer = optim.SGD(model.parameters(), lr=learning_rate, weight_decay=weight_decay)  # Uncomment for SGD

# Training loop with early stopping
consecutive_below_threshold = 0

for epoch in range(n_epochs):
    epoch_loss = 0
    num_batches = 0
    
    for images, labels in train_loader:
        # Move data to GPU if available
        images, labels = images.to(device), labels.to(device)
        
        # Flatten images to 1D (28*28)
        images = images.view(-1, 28*28)
        
        # Forward pass
        outputs = model(images)
        loss = loss_function(outputs, labels)
        
        # Backpropagation
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        
        # Track average loss
        epoch_loss += loss.item()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

    # Check if loss stays below target for multiple epochs
    if avg_loss < target_loss:
        consecutive_below_threshold += 1
        if consecutive_below_threshold >= patience:
            print(f"Early stopping triggered! Loss has been below {target_loss} for {patience} consecutive epochs.")
            break
    else:
        consecutive_below_threshold = 0  # Reset if loss rises again

print("Training over!")
```


    ---------------------------------------------------------------------------

    AttributeError                            Traceback (most recent call last)

    Cell In[3], line 4
          2 import torch.nn as nn
          3 import torch.optim as optim
    ----> 4 from torchvision import datasets, transforms
          5 from torch.utils.data import DataLoader
          7 # Check if GPU is available


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/__init__.py:10
          7 # Don't re-order these, we need to load the _C extension (done when importing
          8 # .extensions) before entering _meta_registrations.
          9 from .extension import _HAS_OPS  # usort:skip
    ---> 10 from torchvision import _meta_registrations, datasets, io, models, ops, transforms, utils  # usort:skip
         12 try:
         13     from .version import __version__  # noqa: F401


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/_meta_registrations.py:25
         20         return fn
         22     return wrapper
    ---> 25 @register_meta("roi_align")
         26 def meta_roi_align(input, rois, spatial_scale, pooled_height, pooled_width, sampling_ratio, aligned):
         27     torch._check(rois.size(1) == 5, lambda: "rois must have shape as Tensor[K, 5]")
         28     torch._check(
         29         input.dtype == rois.dtype,
         30         lambda: (
       (...)     33         ),
         34     )


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/torchvision/_meta_registrations.py:18, in register_meta.<locals>.wrapper(fn)
         17 def wrapper(fn):
    ---> 18     if torchvision.extension._has_ops():
         19         get_meta_lib().impl(getattr(getattr(torch.ops.torchvision, op_name), overload_name), fn)
         20     return fn


    AttributeError: partially initialized module 'torchvision' has no attribute 'extension' (most likely due to a circular import)



```python
import torch

print(f"PyTorch version: {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
       print(f"CUDA version: {torch.version.cuda}")
       print(f"Device count: {torch.cuda.device_count()}")
```

    PyTorch version: 2.6.0
    CUDA available: True
    CUDA version: 12.6
    Device count: 1


## The Big Caveats

The model that we have created and optimzed uses the Pytorch framework which is often favored in academic contexts for its more customizable and less abstracted approach of constructing and training models.

- However in production environments most developers, data scientists included, want a plug and play solution that *can* be customizable but does not force you to customize.

Tensorflow is considered the ideal library to this extent and is written is C++ but has APIs and bindings for a wide variety of languages.

- While **both PyTorch and Tensorflow are written in C++ to be as close to the hardware as possible, Pytorch encourages a more build it yourself approach compared to recent features in TensorFlow.** Tensorflow has tighter integration across the Google ecosystem and in general production technologies and languages like Java and .NET (C#).

So although I don't favor TensorFlow's approach, it's nonetheless important to understand code that is most equivalent to the one above, but in Tensorflow.

Before jumping in lets briefly mention **Keras**!

- If PyTorch allows you to mold your own bricks with no real templates, Keras is Tensorflow's solution to provide brick templates and building blocks to make wholesale models.

### (Tensorflow's) Keras vs PyTorch

- Keras is described as **declarative and high level** where models can be defined and constructed like a LEGO set using its API.

- Keras models are compiled statically prior to training.

- The **.fit() function in Keras can completely abstract away the entire training loop we defined**.

- **Keras is great for prototyping and quick deploys / tests**

- In most use cases you don't need to modify / override the API behavior since the API acts akin to a **Builder pattern and lets you make any model by combining components that are already predefined (its like making a mecha)**.

- However *if* you wanted fine-grained control you could just override `keras.Model`

In contrast, PyTorch is more object-oriented and dynamic, allowing for fine-tuned control and modification of the entire training process. This makes it ideal for researchers and developers who need flexibility in defining custom training logic.

As a TensorFlow developer, you’re not forced to use the Keras API, but its presence greatly improves prototyping and testing workflows. Without Keras, you’d have to manually implement many of the abstractions it provides, making development more complex and time-consuming.

## Tensorflow Equivalent Optimized MNIST Training Loop


```python
import tensorflow as tf
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt

# Check if GPU is available
device = "GPU" if tf.config.list_physical_devices('GPU') else "CPU"
print(f"Using device: {device}")

# Hyperparameters
input_size = 28 * 28
hidden_size = 50
output_size = 10
learning_rate = 0.01
weight_decay = 1e-4  # L2 regularization
n_epochs = 50  # Max epochs
batch_size = 64
target_loss = 0.08  # Stop if loss stays below this
patience = 3  # Consecutive epochs below target before stopping

# Load MNIST dataset
(x_train, y_train), _ = keras.datasets.mnist.load_data()
x_train = x_train.astype(np.float32) / 255.0  # Normalize
x_train = x_train.reshape(-1, input_size)  # Flatten
y_train = y_train.astype(np.int32)  # Ensure labels are integer

# Create a TensorFlow dataset and batch it
train_dataset = tf.data.Dataset.from_tensor_slices((x_train, y_train))
train_dataset = train_dataset.batch(batch_size)

# Define model using the Functional API (more flexible than Sequential)
inputs = keras.Input(shape=(input_size,))
hidden = keras.layers.Dense(hidden_size, activation='relu', kernel_regularizer=keras.regularizers.l2(weight_decay))(inputs)
outputs = keras.layers.Dense(output_size)(hidden)

# Build model
model = keras.Model(inputs=inputs, outputs=outputs)

# Define loss and optimizer
loss_fn = keras.losses.SparseCategoricalCrossentropy(from_logits=True)
optimizer = keras.optimizers.Adam(learning_rate=learning_rate)  # Swap with SGD if needed

# Custom training loop
consecutive_below_threshold = 0  # Track early stopping condition

# Lets also track the loss history
loss_history = []

for epoch in range(n_epochs):
    epoch_loss = 0
    num_batches = 0

    for batch_x, batch_y in train_dataset: # type: ignore
        with tf.GradientTape() as tape:
            logits = model(batch_x, training=True)
            loss = loss_fn(batch_y, logits)

        # Compute gradients and update weights
        grads = tape.gradient(loss, model.trainable_variables)
        optimizer.apply_gradients(zip(grads, model.trainable_variables))

        # Track loss
        epoch_loss += loss.numpy()
        num_batches += 1

    avg_loss = epoch_loss / num_batches
    loss_history.append(avg_loss) # Store the average loss in a history series
    print(f"Epoch [{epoch+1}/{n_epochs}], Loss: {avg_loss:.4f}")

    # Early stopping condition
    if avg_loss < target_loss:
        consecutive_below_threshold += 1
        if consecutive_below_threshold >= patience:
            print(f"Early stopping triggered! Loss has been below {target_loss} for {patience} consecutive epochs.")
            break
    else:
        consecutive_below_threshold = 0  # Reset if loss rises

print("Training over!")

plt.figure(figsize=(8, 5))
plt.plot(loss_history, marker='o')
plt.title('Training Loss Curve')
plt.xlabel('Epoch')
plt.ylabel('Average Loss')
plt.yscale('log') # This line is optional but it makes the y-scale logarithmic (try commenting it and see what happens)
plt.grid(True)
plt.show()

```

    2025-09-11 10:45:27.515776: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.
    2025-09-11 10:45:27.546492: I tensorflow/core/platform/cpu_feature_guard.cc:210] This TensorFlow binary is optimized to use available CPU instructions in performance-critical operations.
    To enable the following instructions: AVX2 AVX512F AVX512_VNNI AVX512_BF16 FMA, in other operations, rebuild TensorFlow with the appropriate compiler flags.
    2025-09-11 10:45:28.313254: I tensorflow/core/util/port.cc:153] oneDNN custom operations are on. You may see slightly different numerical results due to floating-point round-off errors from different computation orders. To turn them off, set the environment variable `TF_ENABLE_ONEDNN_OPTS=0`.


    Using device: GPU


    WARNING: All log messages before absl::InitializeLog() is called are written to STDERR
    I0000 00:00:1757601929.237732   39879 gpu_device.cc:2020] Created device /job:localhost/replica:0/task:0/device:GPU:0 with 8698 MB memory:  -> device: 0, name: NVIDIA GeForce RTX 4070 SUPER, pci bus id: 0000:01:00.0, compute capability: 8.9
    2025-09-11 10:45:38.564171: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


    Epoch [1/50], Loss: 0.2647



    ---------------------------------------------------------------------------

    KeyboardInterrupt                         Traceback (most recent call last)

    Cell In[5], line 60
         58 # Compute gradients and update weights
         59 grads = tape.gradient(loss, model.trainable_variables)
    ---> 60 optimizer.apply_gradients(zip(grads, model.trainable_variables))
         62 # Track loss
         63 epoch_loss += loss.numpy()


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/keras/src/optimizers/base_optimizer.py:463, in BaseOptimizer.apply_gradients(self, grads_and_vars)
        461 def apply_gradients(self, grads_and_vars):
        462     grads, trainable_variables = zip(*grads_and_vars)
    --> 463     self.apply(grads, trainable_variables)
        464     # Return iterations for compat with tf.keras.
        465     return self._iterations


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/keras/src/optimizers/base_optimizer.py:527, in BaseOptimizer.apply(self, grads, trainable_variables)
        524     grads = [g if g is None else g / scale for g in grads]
        526 # Apply gradient updates.
    --> 527 self._backend_apply_gradients(grads, trainable_variables)
        528 # Apply variable constraints after applying gradients.
        529 for variable in trainable_variables:


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/keras/src/optimizers/base_optimizer.py:593, in BaseOptimizer._backend_apply_gradients(self, grads, trainable_variables)
        590     self._apply_weight_decay(trainable_variables)
        592     # Run update step.
    --> 593     self._backend_update_step(
        594         grads, trainable_variables, self.learning_rate
        595     )
        597 if self.use_ema:
        598     self._update_model_variables_moving_average(
        599         self._trainable_variables
        600     )


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/keras/src/backend/tensorflow/optimizer.py:119, in TFOptimizer._backend_update_step(self, grads, trainable_variables, learning_rate)
        114 trainable_variables = [
        115     v.value if isinstance(v, backend.Variable) else v
        116     for v in trainable_variables
        117 ]
        118 grads_and_vars = list(zip(grads, trainable_variables))
    --> 119 grads_and_vars = self._all_reduce_sum_gradients(grads_and_vars)
        120 tf.__internal__.distribute.interim.maybe_merge_call(
        121     self._distributed_tf_update_step,
        122     self._distribution_strategy,
        123     grads_and_vars,
        124     learning_rate,
        125 )


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/keras/src/backend/tensorflow/optimizer.py:172, in TFOptimizer._all_reduce_sum_gradients(self, grads_and_vars)
        170     else:
        171         reduced_with_nones.append((reduced[reduced_pos], v))
    --> 172         reduced_pos += 1
        173 assert reduced_pos == len(reduced), "Failed to add all gradients"
        174 return reduced_with_nones


    KeyboardInterrupt: 



```python
import requests
import csv
import io

def fetch_and_print_grid(url):
    # Fetch the document (export as CSV is cleaner than HTML)
    if "export" not in url:
        # convert doc URL to CSV export
        url = url.replace("/edit", "/export?format=csv")

    response = requests.get(url)
    response.raise_for_status()

    # Parse CSV
    reader = csv.reader(io.StringIO(response.text))
    header = next(reader)  # skip header row
    data = []
    for row in reader:
        if len(row) == 3:
            x, char, y = row
            data.append((int(x.strip()), char.strip(), int(y.strip())))

    # Get bounds
    max_x = max(x for x, _, _ in data)
    max_y = max(y for _, _, y in data)

    # Build grid
    grid = [[" " for _ in range(max_x + 1)] for x in range(max_y + 1)]
    for x, char, y in data:
        grid[y][x] = char

    # Print grid
    for row in grid:
        print("".join(row))

# Example usage:
fetch_and_print_grid("")
```


    ---------------------------------------------------------------------------

    HTTPError                                 Traceback (most recent call last)

    Cell In[5], line 37
         34         print("".join(row))
         36 # Example usage:
    ---> 37 fetch_and_print_grid("https://docs.google.com/document/d/e/2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9DIz-vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub")


    Cell In[5], line 12, in fetch_and_print_grid(url)
          9     url = url.replace("/edit", "/export?format=csv")
         11 response = requests.get(url)
    ---> 12 response.raise_for_status()
         14 # Parse CSV
         15 reader = csv.reader(io.StringIO(response.text))


    File /opt/miniforge/envs/mlshit/lib/python3.12/site-packages/requests/models.py:1026, in Response.raise_for_status(self)
       1021     http_error_msg = (
       1022         f"{self.status_code} Server Error: {reason} for url: {self.url}"
       1023     )
       1025 if http_error_msg:
    -> 1026     raise HTTPError(http_error_msg, response=self)


    HTTPError: 404 Client Error: Not Found for url: https://docs.google.com/document/d/e/2PACX-1vRPzbNQcx5UriHSbZ-9vmsTow_R6RRe7eyAU60xIF9DIz-vaHiHNO2TKgDi7jy4ZpTpNqM7EvEcfr_p/pub



```python
# The code below tests the trained model!

# Load test data (we skipped it earlier with the "_")
_, (x_test, y_test) = keras.datasets.mnist.load_data()

# Preprocess test data
x_test = x_test.astype(np.float32) / 255.0  # Normalize
x_test = x_test.reshape(-1, input_size)     # Flatten
y_test = y_test.astype(np.int32)

# Create TensorFlow dataset
test_dataset = tf.data.Dataset.from_tensor_slices((x_test, y_test)).batch(batch_size)

# Evaluate accuracy
correct = 0
total = 0

for batch_x, batch_y in test_dataset:
    logits = model(batch_x, training=False)
    preds = tf.argmax(logits, axis=1, output_type=tf.int32)
    correct += tf.reduce_sum(tf.cast(preds == batch_y, tf.int32)).numpy()
    total += batch_x.shape[0]

accuracy = correct / total
print(f"Test accuracy: {accuracy:.4f}")
```

    Test accuracy: 0.9547


    2025-09-11 10:28:49.680967: I tensorflow/core/framework/local_rendezvous.cc:407] Local rendezvous is aborting with status: OUT_OF_RANGE: End of sequence


## Introduction to Convolutional Neural Networks

In classical multi-layer perceptron (MLP) networks, our inputs must be *flat*: each input neuron receives a single feature without any intrinsic understanding of how that feature relates to the others.

- For inputs where the data points are entirely independent or unordered, this may be perfectly acceptable.

However, in the case of images, the relationship between two pixels depends heavily on their relative positions (their **locality** within the image). How can we design a neural network that not only accepts image data but also preserves and leverages this information about pixel positions?

Consider a simple grayscale image of **256×256** pixels:
- This would give us **196,608 input features** (256 × 256 = 65,536; add 3 channels for RGB → 196,608).
- If we connected each input to **1,000 neurons in the first hidden layer**, that would mean nearly **197 million parameters** just for that one layer!

Humans don’t identify objects by examining pixels individually. Instead, we recognize patterns that form meaningful structures across different regions of an image.

Let’s take a **dog** as an example:
- We identify a dog by the combination of its **head**, **body**, and **tail**.
- The head might have a long snout, bead-like eyes, floppy ears, and sometimes a tongue sticking out.
- The body might show soft fur, an oblong shape, and typical color patterns like black, brown, gold, or white — plus four legs.
- The tail might be straight or curly, but is generally longer than that of smaller animals like rabbits.

We can break this down further:
- Each of these parts — head, body, tail — can be described in terms of smaller features (like the shape of an ear or the curve of a tail).
- Eventually, we might say that *specific patterns of pixels* define what we recognize as a dog.

**The key idea:**  

Humans intuitively decompose objects into patterns and sub-patterns.
- **Convolutional Neural Networks (CNNs)** attempt to *reverse engineer* this intuition.
- CNNs scan the image for small, local patterns, then combine them hierarchically into more abstract features — much like how our eyes and brain work together to recognize objects.
- The idea is to **look through an image in chunks to see how close that chunk of an image is to a feature you're looking for**, in the process you end up creating something called a **feature map**.

<img src="/notebooks/media/CNN_visualization.gif" width="500px">

---

### Kernels, Filters, and Features

To do this, CNNs rely on a few key components:

- **Kernels** (or **filters**): small, learnable matrices that slide (or *convolve*) across the image. Each kernel scans for a specific pattern, like an edge, a texture, or a corner.
- **Features**: the output of applying these kernels to the input image. You can think of these as the *maps* of where particular patterns were detected in the image.

The filter defines what we are looking for, for simple features such as corners or edges, our filter could be pretty simple and small. 

Here is an image which shows some simple 3 by 3 filters.

<img src="/notebooks/media/first_level_features_in_CNN.webp" width="500px">

**The process by which we sweep our filter across the image is called convolution (hence the name Convolutional Neural Networks)**

- When the kernel is superimposed on a part of the image, the mathematical operation used to generate a singular number (feature) out of the portion of the image and the applied kernel is the **dot product**.

- - We compute **element-wise products between the kernel values and the corresponding patch pixel values**.
- - We **sum these products to get a single scalar at that location**.
- - This is mathematically equivalent to taking the dot product of two vectors: the flattened kernel and the flattened patch.

Suppose two matrices:

The kernel matrix:

$$
\begin{bmatrix}
-1 & -1 & -1 \\
0 & 0 & 0 \\
1 & 1 & 1 \\
\end{bmatrix}
$$

The image patch's matrix:

$$
\begin{bmatrix}
1 & 2 & 3 \\
4 & 5 & 6 \\
7 & 8 & 9 \\
\end{bmatrix}
$$

Our singular feature would be calculated as: $-1 \cdot 1 + -1 \cdot 2 + -1 \cdot 3 + 0 \cdot 4 + 0 \cdot 5 + 0 \cdot 6 + 1 \cdot 7 + 1 \cdot 8 + 1 \cdot 9 = 18$

Of course in practice the pixel values in the kernel and the picture may range from 0 to 1 if the image has a single, alpha channel. However for traditional images with multiple channels, multiplying the values in one pixel by that of a kernel may have much more nuanced effects.

- In the prior example, the kernel was filtering for a **horizontal edge at the bottom**, it accomplished this by giving a positive weight to the row of pixels at the bottom of the window, giving no (0) weight to the row in the middle of the window, and negative weights to the row of pixels at the top of the window.

- - **Alternatively the kernel could have given 0 weight to both of the top two rows and positive weight to the bottom row. But why would this be less effective?**

<details>
<summary> Click for Answer! </summary>
<p>
    An alternative kernel that gives zero weight to the top two rows and positive weight only to the bottom row would not detect edges as effectively. 
    Instead of highlighting <em>changes</em> between regions, it would simply emphasize areas where the bottom row has high pixel values — 
    even if the top rows are similar. <b>Effective edge detection relies on comparing regions looking for sharp differences between areas 
    and this is why kernels are designed to include both positive and negative weights.</b>
</p>
</details>

Our kernel (or filter) acts as a weighted sum function, where the adjustable parameters are the values within the kernel itself, and the data is the patch of the image the kernel is applied to.

---

### Key points

- When training a convolutional neural network, we tune each kernel on the same layer independently. **Each kernel acts as its own trainable function, scanning the image and producing a feature map, which will be combined with other features in the same level of the network.**

- **However, only the first layer of kernels interacts directly with the image. Higher-level kernels act on the abstractions captured in the feature maps generated by lower-level kernels, allowing the network to build increasingly complex representations.**

**The final dog probability is computed from a weighted combination of all these aggregated high-level features, not just one map.**

Here are a few key questions to consider as we talk about CNNs in later lessons:

- Is there an art to how we tell the kernel to scan over the image? Should there be less or no overlap in each patch it scans? 
- - How does this affect the size of the output feature map? 
- - What if an instance of "scanning" and applying the kernel results in the kernel going out of the image's bounds?

- How do we downsize the resulting feature maps if they become too large or cumbersome?

- Do the steps describe so far prevent CNNs from having issues with representing non-linear relationships, if not, how can we address the issue?

- Even if we can detect low-level features, how exactly are these combined to represent more abstract patterns or objects?

- Is there an intuitive, visual way to see what a kernel is “looking at” over the image — like a heatmap that shows its focus across the input?
