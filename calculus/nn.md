# Neural Network with Two Layers

Welcome to your week three programming assignment. You are ready to build a neural network with two layers and train it to solve a classification problem. 

**After this assignment, you will be able to:**

- Implement a neural network with two layers to a classification problem
- Implement forward propagation using matrix multiplication
- Perform backward propagation

# Table of Contents

- [ 1 - Classification Problem](#1)
- [ 2 - Neural Network Model with Two Layers](#2)
  - [ 2.1 - Neural Network Model with Two Layers for a Single Training Example](#2.1)
  - [ 2.2 - Neural Network Model with Two Layers for Multiple Training Examples](#2.2)
  - [ 2.3 - Cost Function and Training](#2.3)
  - [ 2.4 - Dataset](#2.4)
  - [ 2.5 - Define Activation Function](#2.5)
    - [ Exercise 1](#ex01)
- [ 3 - Implementation of the Neural Network Model with Two Layers](#3)
  - [ 3.1 - Defining the Neural Network Structure](#3.1)
    - [ Exercise 2](#ex02)
  - [ 3.2 - Initialize the Model's Parameters](#3.2)
    - [ Exercise 3](#ex03)
  - [ 3.3 - The Loop](#3.3)
    - [ Exercise 4](#ex04)
    - [ Exercise 5](#ex05)
    - [ Exercise 6](#ex06)
  - [ 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model()](#3.4)
    - [ Exercise 7](#ex07)
    - [ Exercise 8](#ex08)
- [ 4 - Optional: Other Dataset](#4)

## Packages

First, import all the packages you will need during this assignment.


```python
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors
# A function to create a dataset.
from sklearn.datasets import make_blobs

# Output of plotting commands is displayed inline within the Jupyter notebook.
%matplotlib inline 

# Set a seed so that the results are consistent.
np.random.seed(3)
```

Load the unit tests defined for this notebook.


```python
import w3_unittest
```

<a name='1'></a>
## 1 - Classification Problem

In one of the labs this week, you trained a neural network with a single perceptron, performing forward and backward propagation. That simple structure was enough to solve a "linear" classification problem - finding a straight line in a plane that would serve as a decision boundary to separate two classes.

Imagine that now you have a more complicated problem: you still have two classes, but one line will not be enough to separate them.


```python
fig, ax = plt.subplots()
xmin, xmax = -0.2, 1.4
x_line = np.arange(xmin, xmax, 0.1)
# Data points (observations) from two classes.
ax.scatter(0, 0, color="r")
ax.scatter(0, 1, color="b")
ax.scatter(1, 0, color="b")
ax.scatter(1, 1, color="r")
ax.set_xlim([xmin, xmax])
ax.set_ylim([-0.1, 1.1])
ax.set_xlabel('$x_1$')
ax.set_ylabel('$x_2$')
# Example of the lines which can be used as a decision boundary to separate two classes.
ax.plot(x_line, -1 * x_line + 1.5, color="black")
ax.plot(x_line, -1 * x_line + 0.5, color="black")
plt.plot()
```




    []




    
![png](output_8_1.png)
    


This logic can appear in many applications. For example, if you train a model to predict whether you should buy a house knowing its size and the year it was built. A big new house will not be affordable, while a small old house will not be worth buying. So, you might be interested in either a big old house, or a small new house.

The one perceptron neural network is not enough to solve such classification problem. Let's look at how you can adjust that model to find the solution.

In the plot above, two lines can serve as a decision boundary. Your intuition might tell you that you should also increase the number of perceptrons. And that is absolutely right! You need to feed your data points (coordinates $x_1$, $x_2$) into two nodes separately and then unify them somehow with another one to make a decision. 

Now let's figure out the details, build and train your first multi-layer neural network!

<a name='2'></a>
## 2 - Neural Network Model with Two Layers

<a name='2.1'></a>
### 2.1 - Neural Network Model with Two Layers for a Single Training Example

<img src="images/nn_model_2_layers.png" style="width:1000px;">

The input and output layers of the neural network are the same as for one perceptron model, but there is a **hidden layer** now in between them. The training examples $x^{(i)}=\begin{bmatrix}x_1^{(i)} \\ x_2^{(i)}\end{bmatrix}$ from the input layer of size $n_x = 2$ are first fed into the hidden layer of size $n_h = 2$. They are simultaneously fed into the first perceptron with weights $W_1^{[1]}=\begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]}\end{bmatrix}$, bias  $b_1^{[1]}$; and into the second perceptron with weights $W_2^{[1]}=\begin{bmatrix}w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$, bias $b_2^{[1]}$. The integer in the square brackets $^{[1]}$ denotes the layer number, because there are two layers now with their own parameters and outputs, which need to be distinguished. 

\begin{align}
z_1^{[1](i)} &= w_{1,1}^{[1]} x_1^{(i)} + w_{2,1}^{[1]} x_2^{(i)} + b_1^{[1]} = W_1^{[1]}x^{(i)} + b_1^{[1]},\\
z_2^{[1](i)} &= w_{1,2}^{[1]} x_1^{(i)} + w_{2,2}^{[1]} x_2^{(i)} + b_2^{[1]} = W_2^{[1]}x^{(i)} + b_2^{[1]}.\tag{1}
\end{align}

These expressions for one training example $x^{(i)}$ can be rewritten in a matrix form :

$$z^{[1](i)} = W^{[1]} x^{(i)} + b^{[1]},\tag{2}$$

where 

&emsp; &emsp; $z^{[1](i)} = \begin{bmatrix}z_1^{[1](i)} \\ z_2^{[1](i)}\end{bmatrix}$ is vector of size $\left(n_h \times 1\right) = \left(2 \times 1\right)$; 

&emsp; &emsp; $W^{[1]} = \begin{bmatrix}W_1^{[1]} \\ W_2^{[1]}\end{bmatrix} = 
\begin{bmatrix}w_{1,1}^{[1]} & w_{2,1}^{[1]} \\ w_{1,2}^{[1]} & w_{2,2}^{[1]}\end{bmatrix}$ is matrix of size $\left(n_h \times n_x\right) = \left(2 \times 2\right)$;

&emsp; &emsp; $b^{[1]} = \begin{bmatrix}b_1^{[1]} \\ b_2^{[1]}\end{bmatrix}$ is vector of size $\left(n_h \times 1\right) = \left(2 \times 1\right)$.

Next, the hidden layer activation function needs to be applied for each of the elements in the vector $z^{[1](i)}$. Various activation functions can be used here and in this model you will take the sigmoid function $\sigma\left(x\right) = \frac{1}{1 + e^{-x}}$. Remember that its derivative is $\frac{d\sigma}{dx} = \sigma\left(x\right)\left(1-\sigma\left(x\right)\right)$. The output of the hidden layer is a vector of size $\left(n_h \times 1\right) = \left(2 \times 1\right)$:

$$a^{[1](i)} = \sigma\left(z^{[1](i)}\right) = 
\begin{bmatrix}\sigma\left(z_1^{[1](i)}\right) \\ \sigma\left(z_2^{[1](i)}\right)\end{bmatrix}.\tag{3}$$

Then the hidden layer output gets fed into the output layer of size $n_y = 1$. This was covered in the previous lab, the only difference are: $a^{[1](i)}$ is taken instead of $x^{(i)}$ and layer notation $^{[2]}$ appears to identify all parameters and outputs:

$$z^{[2](i)} = w_1^{[2]} a_1^{[1](i)} + w_2^{[2]} a_2^{[1](i)} + b^{[2]}= W^{[2]} a^{[1](i)} + b^{[2]},\tag{4}$$

&emsp; &emsp; $z^{[2](i)}$ and $b^{[2]}$ are scalars for this model, as $\left(n_y \times 1\right) = \left(1 \times 1\right)$; 

&emsp; &emsp; $W^{[2]} = \begin{bmatrix}w_1^{[2]} & w_2^{[2]}\end{bmatrix}$ is vector of size $\left(n_y \times n_h\right) = \left(1 \times 2\right)$.

Finally, the same sigmoid function is used as the output layer activation function:

$$a^{[2](i)} = \sigma\left(z^{[2](i)}\right).\tag{5}$$

Mathematically the two layer neural network model for each training example $x^{(i)}$ can be written with the expressions $(2) - (5)$. Let's rewrite them next to each other for convenience:

\begin{align}
z^{[1](i)} &= W^{[1]} x^{(i)} + b^{[1]},\\
a^{[1](i)} &= \sigma\left(z^{[1](i)}\right),\\
z^{[2](i)} &= W^{[2]} a^{[1](i)} + b^{[2]},\\
a^{[2](i)} &= \sigma\left(z^{[2](i)}\right).\\
\tag{6}
\end{align}

Note, that all of the parameters to be trained in the model are without $^{(i)}$ index - they are independent on the input data.

Finally, the predictions for some example $x^{(i)}$ can be made taking the output $a^{[2](i)}$ and calculating $\hat{y}$ as: $\hat{y} = \begin{cases} 1 & \mbox{if } a^{[2](i)} > 0.5, \\ 0 & \mbox{otherwise }. \end{cases}$.

<a name='2.2'></a>
### 2.2 - Neural Network Model with Two Layers for Multiple Training Examples

Similarly to the single perceptron model, $m$ training examples can be organised in a matrix $X$ of a shape ($2 \times m$), putting $x^{(i)}$ into columns. Then the model $(6)$ can be rewritten in terms of matrix multiplications:

\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]},\\
A^{[1]} &= \sigma\left(Z^{[1]}\right),\\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]},\\
A^{[2]} &= \sigma\left(Z^{[2]}\right),\\
\tag{7}
\end{align}

where $b^{[1]}$ is broadcasted to the matrix of size $\left(n_h \times m\right) = \left(2 \times m\right)$ and $b^{[2]}$ to the vector of size $\left(n_y \times m\right) = \left(1 \times m\right)$. It would be a good exercise for you to have a look at the expressions $(7)$ and check that sizes of the matrices will actually match to perform required multiplications.

You have derived expressions to perform forward propagation. Time to evaluate your model and train it.

<a name='2.3'></a>
### 2.3 - Cost Function and Training

For the evaluation of this simple neural network you can use the same cost function as for the single perceptron case - log loss function. Originally initialized weights were just some random values, now you need to perform training of the model: find such set of parameters $W^{[1]}$, $b^{[1]}$, $W^{[2]}$, $b^{[2]}$, that will minimize the cost function.

Like in the previous example of a single perceptron neural network, the cost function can be written as:

$$\mathcal{L}\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right) = \frac{1}{m}\sum_{i=1}^{m} L\left(W^{[1]}, b^{[1]}, W^{[2]}, b^{[2]}\right) =  \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small - y^{(i)}\log\left(a^{[2](i)}\right) - (1-y^{(i)})\log\left(1- a^{[2](i)}\right)  \large  \right), \small\tag{8}$$

where $y^{(i)} \in \{0,1\}$ are the original labels and $a^{[2](i)}$ are the continuous output values of the forward propagation step (elements of array $A^{[2]}$).

To minimize it, you can use gradient descent, updating the parameters with the following expressions:

\begin{align}
W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\\
b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\\
W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\\
b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} },\\
\tag{9}
\end{align}

where $\alpha$ is the learning rate.


To perform training of the model you need to calculate now $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$, $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$, $\frac{\partial \mathcal{L} }{ \partial W^{[2]}}$, $\frac{\partial \mathcal{L} }{ \partial b^{[2]}}$. 

Let's start from the end of the neural network. You can rewrite here the corresponding expressions for $\frac{\partial \mathcal{L} }{ \partial W }$ and $\frac{\partial \mathcal{L} }{ \partial b }$ from the single perceptron neural network:

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W } &= 
\frac{1}{m}\left(A-Y\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b } &= 
\frac{1}{m}\left(A-Y\right)\mathbf{1},\\
\end{align}

where $\mathbf{1}$ is just a ($m \times 1$) vector of ones. Your one perceptron is in the second layer now, so $W$ will be exchanged with $W^{[2]}$, $b$ with $b^{[2]}$, $A$ with $A^{[2]}$, $X$ with $A^{[1]}$:

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1}.\\
\tag{10}
\end{align}


Let's now find $\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = 
\begin{bmatrix}
\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,1}^{[1]}} \\
\frac{\partial \mathcal{L} }{ \partial w_{1,2}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,2}^{[1]}} \end{bmatrix}$. It was shown in the videos that $$\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}}=\frac{1}{m}\sum_{i=1}^{m} \left( 
\left(a^{[2](i)} - y^{(i)}\right) 
w_1^{[2]} 
\left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_1^{(i)}\right)\tag{11}$$

If you do this accurately for each of the elements $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$, you will get the following matrix:

$$\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \begin{bmatrix}
\frac{\partial \mathcal{L} }{ \partial w_{1,1}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,1}^{[1]}} \\
\frac{\partial \mathcal{L} }{ \partial w_{1,2}^{[1]}} & \frac{\partial \mathcal{L} }{ \partial w_{2,2}^{[1]}} \end{bmatrix}$$
$$= \frac{1}{m}\begin{bmatrix}
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_2^{(i)}\right)  \\
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_2^{(i)}\right)\end{bmatrix}\tag{12}$$

Looking at this, you can notice that all terms and indices somehow are very consistent, so it all can be unified into a matrix form. And that's true! $\left(W^{[2]}\right)^T = \begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix}$ of size $\left(n_h \times n_y\right) = \left(2 \times 1\right)$ can be multiplied with the vector $A^{[2]} - Y$ of size $\left(n_y \times m\right) = \left(1 \times m\right)$, resulting in a matrix of size $\left(n_h \times m\right) = \left(2 \times m\right)$:

$$\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)=
\begin{bmatrix}w_1^{[2]} \\ w_2^{[2]}\end{bmatrix}
\begin{bmatrix}\left(a^{[2](1)} - y^{(1)}\right) &  \cdots & \left(a^{[2](m)} - y^{(m)}\right)\end{bmatrix}
=\begin{bmatrix}
\left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]} & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]} \\
\left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]} & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \end{bmatrix}$$.

Now taking matrix $A^{[1]}$ of the same size $\left(n_h \times m\right) = \left(2 \times m\right)$,

$$A^{[1]}
=\begin{bmatrix}
a_1^{[1](1)} & \cdots & a_1^{[1](m)} \\
a_2^{[1](1)} & \cdots & a_2^{[1](m)} \end{bmatrix},$$

you can calculate:

$$A^{[1]}\cdot\left(1-A^{[1]}\right)
=\begin{bmatrix}
a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right) & \cdots & a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right) \\
a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right) & \cdots & a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right) \end{bmatrix},$$

where "$\cdot$" denotes **element by element** multiplication.

With the element by element multiplication,

$$\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)=\begin{bmatrix}
\left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]}\left(a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]}\left(a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right)\right) \\
\left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]}\left(a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \left(a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right)\right) \end{bmatrix}.$$

If you perform matrix multiplication with $X^T$ of size $\left(m \times n_x\right) = \left(m \times 2\right)$, you will get matrix of size $\left(n_h \times n_x\right) = \left(2 \times 2\right)$:

$$\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T = 
\begin{bmatrix}
\left(a^{[2](1)} - y^{(1)}\right) w_1^{[2]}\left(a_1^{[1](1)}\left(1 - a_1^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_1^{[2]}\left(a_1^{[1](m)}\left(1 - a_1^{[1](m)}\right)\right) \\
\left(a^{[2](1)} - y^{(1)}\right) w_2^{[2]}\left(a_2^{[1](1)}\left(1 - a_2^{[1](1)}\right)\right) & \cdots & \left(a^{[2](m)} - y^{(m)}\right) w_2^{[2]} \left(a_2^{[1](m)}\left(1 - a_2^{[1](m)}\right)\right) \end{bmatrix}
\begin{bmatrix}
x_1^{(1)} & x_2^{(1)} \\
\cdots & \cdots \\
x_1^{(m)} & x_2^{(m)}
\end{bmatrix}$$
$$=\begin{bmatrix}
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1 - a_1^{[1](i)}\right) \right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_1^{[2]} \left(a_1^{[1](i)}\left(1-a_1^{[1](i)}\right)\right)
x_2^{(i)}\right)  \\
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_1^{(i)}\right) & 
\sum_{i=1}^{m} \left( \left(a^{[2](i)} - y^{(i)}\right) w_2^{[2]} \left(a_2^{[1](i)}\left(1-a_2^{[1](i)}\right)\right)
x_2^{(i)}\right)\end{bmatrix}$$

This is exactly like in the expression $(12)$! So, $\frac{\partial \mathcal{L} }{ \partial W^{[1]}}$ can be written as a mixture of multiplications:

$$\frac{\partial \mathcal{L} }{ \partial W^{[1]}} = \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T\tag{13},$$

where "$\cdot$" denotes element by element multiplications.

Vector $\frac{\partial \mathcal{L} }{ \partial b^{[1]}}$ can be found very similarly, but the last terms in the chain rule will be equal to $1$, i.e. $\frac{\partial z_1^{[1](i)}}{ \partial b_1^{[1]}} = 1$. Thus,

$$\frac{\partial \mathcal{L} }{ \partial b^{[1]}} = \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1},\tag{14}$$

where $\mathbf{1}$ is a ($m \times 1$) vector of ones.

Expressions $(10)$, $(13)$ and $(14)$ can be used for the parameters update $(9)$ performing backward propagation:

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\\
\frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1},\\
\tag{15}
\end{align}

where $\mathbf{1}$ is a ($m \times 1$) vector of ones.

So, to understand deeply and properly how neural networks perform and get trained, **you do need knowledge of linear algebra and calculus joined together**! But do not worry! All together it is not that scary if you do it step by step accurately with understanding of maths.

Time to implement this all in the code!

<a name='2.2'></a>
### 2.2 - Dataset

First, let's get the dataset you will work on. The following code will create $m=2000$ data points $(x_1, x_2)$ and save them in the `NumPy` array `X` of a shape $(2 \times m)$ (in the columns of the array). The labels ($0$: blue, $1$: red) will be saved in the `NumPy` array `Y` of a shape $(1 \times m)$.


```python
m = 2000
samples, labels = make_blobs(n_samples=m, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0) | (labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 0
X = np.transpose(samples)
Y = labels.reshape((1, m))

plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

print ('The shape of X is: ' + str(X.shape))
print ('The shape of Y is: ' + str(Y.shape))
print ('I have m = %d training examples!' % (m))
```

    The shape of X is: (2, 2000)
    The shape of Y is: (1, 2000)
    I have m = 2000 training examples!



    
![png](output_27_1.png)
    


<a name='2.3'></a>
### 2.3 - Define Activation Function

<a name='ex01'></a>
### Exercise 1

Define sigmoid activation function $\sigma\left(z\right) =\frac{1}{1+e^{-z}} $.


```python
def sigmoid(z):
    ### START CODE HERE ### (~ 1 line of code)
    res = 1 / (1 + np.exp(-z))
    ### END CODE HERE ###
    
    return res
```


```python
print("sigmoid(-2) = " + str(sigmoid(-2)))
print("sigmoid(0) = " + str(sigmoid(0)))
print("sigmoid(3.5) = " + str(sigmoid(3.5)))
```

    sigmoid(-2) = 0.11920292202211755
    sigmoid(0) = 0.5
    sigmoid(3.5) = 0.9706877692486436


##### __Expected Output__

Note: the values may vary in the last decimal places.

```Python
sigmoid(-2) = 0.11920292202211755
sigmoid(0) = 0.5
sigmoid(3.5) = 0.9706877692486436
```


```python
w3_unittest.test_sigmoid(sigmoid)
```

    [92m All tests passed


<a name='3'></a>
## 3 - Implementation of the Neural Network Model with Two Layers

<a name='3.1'></a>
### 3.1 - Defining the Neural Network Structure

<a name='ex02'></a>
### Exercise 2

Define three variables:
- `n_x`: the size of the input layer
- `n_h`: the size of the hidden layer (set it equal to 2 for now)
- `n_y`: the size of the output layer

<details>    
<summary>
    <font size="3" color="darkgreen"><b>Hint</b></font>
</summary>
<p>
<ul>
    Use shapes of X and Y to find n_x and n_y:
    <li>the size of the input layer n_x equals to the size of the input vectors placed in the columns of the array X,</li>
    <li>the outpus for each of the data point will be saved in the columns of the the array Y.</li>
</ul>
</p>


```python
# GRADED FUNCTION: layer_sizes

def layer_sizes(X, Y):
    """
    Arguments:
    X -- input dataset of shape (input size, number of examples)
    Y -- labels of shape (output size, number of examples)
    
    Returns:
    n_x -- the size of the input layer
    n_h -- the size of the hidden layer
    n_y -- the size of the output layer
    """
    ### START CODE HERE ### (~ 3 lines of code)
    # Size of input layer.
    n_x = X.shape[0]
    # Size of hidden layer.
    n_h = 2
    # Size of output layer.
    n_y = Y.shape[0]
    ### END CODE HERE ###
    return (n_x, n_h, n_y)
```


```python
(n_x, n_h, n_y) = layer_sizes(X, Y)
print("The size of the input layer is: n_x = " + str(n_x))
print("The size of the hidden layer is: n_h = " + str(n_h))
print("The size of the output layer is: n_y = " + str(n_y))
```

    The size of the input layer is: n_x = 2
    The size of the hidden layer is: n_h = 2
    The size of the output layer is: n_y = 1


##### __Expected Output__

```Python
The size of the input layer is: n_x = 2
The size of the hidden layer is: n_h = 2
The size of the output layer is: n_y = 1
```


```python
w3_unittest.test_layer_sizes(layer_sizes)
```

    [92m All tests passed


<a name='3.2'></a>
### 3.2 - Initialize the Model's Parameters

<a name='ex03'></a>
### Exercise 3

Implement the function `initialize_parameters()`.

**Instructions**:
- Make sure your parameters' sizes are right. Refer to the neural network figure above if needed.
- You will initialize the weights matrix with random values. 
    - Use: `np.random.randn(a,b) * 0.01` to randomly initialize a matrix of shape (a,b).
- You will initialize the bias vector as zeros. 
    - Use: `np.zeros((a,b))` to initialize a matrix of shape (a,b) with zeros.


```python
# GRADED FUNCTION: initialize_parameters

def initialize_parameters(n_x, n_h, n_y):
    """
    Argument:
    n_x -- size of the input layer
    n_h -- size of the hidden layer
    n_y -- size of the output layer
    
    Returns:
    params -- python dictionary containing your parameters:
                    W1 -- weight matrix of shape (n_h, n_x)
                    b1 -- bias vector of shape (n_h, 1)
                    W2 -- weight matrix of shape (n_y, n_h)
                    b2 -- bias vector of shape (n_y, 1)
    """
    
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = np.random.randn(n_h, n_x) * 0.01
    b1 = np.zeros((n_h, 1))
    W2 = np.random.randn(n_y, n_h)
    b2 = np.zeros((n_y, 1))
    ### END CODE HERE ###
    
    assert (W1.shape == (n_h, n_x))
    assert (b1.shape == (n_h, 1))
    assert (W2.shape == (n_y, n_h))
    assert (b2.shape == (n_y, 1))
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```


```python
parameters = initialize_parameters(n_x, n_h, n_y)

print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))
```

    W1 = [[ 0.01788628  0.0043651 ]
     [ 0.00096497 -0.01863493]]
    b1 = [[0.]
     [0.]]
    W2 = [[-0.2773882  -0.35475898]]
    b2 = [[0.]]


##### __Expected Output__ 
Note: the elements of the arrays W1 and W2 maybe be different due to random initialization. You can try to restart the kernel to get the same values.

```Python
W1 = [[ 0.01788628  0.0043651 ]
 [ 0.00096497 -0.01863493]]
b1 = [[0.]
 [0.]]
W2 = [[-0.00277388 -0.00354759]]
b2 = [[0.]]
```


```python
# Note: 
# Actual values are not checked here in the unit tests (due to random initialization).
w3_unittest.test_initialize_parameters(initialize_parameters)
```

    [92m All tests passed


<a name='3.3'></a>
### 3.3 - The Loop

<a name='ex04'></a>
### Exercise 4

Implement `forward_propagation()`.

**Instructions**:
- Look above at the mathematical representation $(7)$ of your classifier (section [2.2](#2.2)):
\begin{align}
Z^{[1]} &= W^{[1]} X + b^{[1]},\\
A^{[1]} &= \sigma\left(Z^{[1]}\right),\\
Z^{[2]} &= W^{[2]} A^{[1]} + b^{[2]},\\
A^{[2]} &= \sigma\left(Z^{[2]}\right).\\
\end{align}
- The steps you have to implement are:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
    2. Implement Forward Propagation. Compute `Z1` multiplying matrices `W1`, `X` and adding vector `b1`. Then find `A1` using the `sigmoid` activation function. Perform similar computations for `Z2` and `A2`.


```python
# GRADED FUNCTION: forward_propagation

def forward_propagation(X, parameters):
    """
    Argument:
    X -- input data of size (n_x, m)
    parameters -- python dictionary containing your parameters (output of initialization function)
    
    Returns:
    A2 -- the sigmoid output of the second activation
    cache -- python dictionary containing Z1, A1, Z2, A2 
    (that simplifies the calculations in the back propagation step)
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Implement forward propagation to calculate A2.
    ### START CODE HERE ### (~ 4 lines of code)
    Z1 = np.matmul(W1, X) + b1
    A1 = sigmoid(Z1)
    Z2 = np.matmul(W2, A1) + b2
    A2 = sigmoid(Z2)
    ### END CODE HERE ###
    
    assert(A2.shape == (n_y, X.shape[1]))

    cache = {"Z1": Z1,
             "A1": A1,
             "Z2": Z2,
             "A2": A2}
    
    return A2, cache
```


```python
A2, cache = forward_propagation(X, parameters)

print(A2)
```

    [[0.42082836 0.42285507 0.42186862 ... 0.42186021 0.42169268 0.42132431]]


##### __Expected Output__ 
Note: the elements of the array A2 maybe be different depending on the initial parameters. If you would like to get exactly the same output, try to restart the Kernel and rerun the notebook.

```Python
[[0.49920157 0.49922234 0.49921223 ... 0.49921215 0.49921043 0.49920665]]
```


```python
# Note: 
# Actual values are not checked here in the unit tests (due to random initialization).
w3_unittest.test_forward_propagation(forward_propagation)
```

    [92m All tests passed


Remember, that your weights were just initialized with some random values, so the model has not been trained yet. 

<a name='ex05'></a>
### Exercise 5

Define a cost function $(8)$ which will be used to train the model:

$$\mathcal{L}\left(W, b\right)  = \frac{1}{m}\sum_{i=1}^{m}  \large\left(\small - y^{(i)}\log\left(a^{(i)}\right) - (1-y^{(i)})\log\left(1- a^{(i)}\right)  \large  \right) \small.$$


```python
def compute_cost(A2, Y):
    """
    Computes the cost function as a log loss
    
    Arguments:
    A2 -- The output of the neural network of shape (1, number of examples)
    Y -- "true" labels vector of shape (1, number of examples)
    
    Returns:
    cost -- log loss
    
    """
    # Number of examples.
    m = Y.shape[1]
    
    ### START CODE HERE ### (~ 2 lines of code)
    logloss = np.sum(-Y * np.log(A2) - (1 - Y) * np.log(1 - A2)) / m
    cost = logloss
    ### END CODE HERE ###

    assert(isinstance(cost, float))
    
    return cost
```


```python
print("cost = " + str(compute_cost(A2, Y)))
```

    cost = 0.7054321721893957


##### __Expected Output__ 
Note: the elements of the arrays W1 and W2 maybe be different!

```Python
cost = 0.6931477703826823
```


```python
# Note: 
# Actual values are not checked here in the unit tests (due to random initialization).
w3_unittest.test_compute_cost(compute_cost, A2)
```

    Test case "default_check". Wrong output of compute_cost. 
    	Expected: 
    0.6931477703826823
    	Got: 
    0.7054321721893957
    [92m 1  Tests passed
    [91m 1  Tests failed


Calculate partial derivatives as shown in $(15)$:

\begin{align}
\frac{\partial \mathcal{L} }{ \partial W^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\left(A^{[1]}\right)^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[2]} } &= 
\frac{1}{m}\left(A^{[2]}-Y\right)\mathbf{1},\\
\frac{\partial \mathcal{L} }{ \partial W^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)X^T,\\
\frac{\partial \mathcal{L} }{ \partial b^{[1]}} &= \frac{1}{m}\left(\left(W^{[2]}\right)^T \left(A^{[2]} - Y\right)\cdot \left(A^{[1]}\cdot\left(1-A^{[1]}\right)\right)\right)\mathbf{1}.\\
\end{align}


```python
def backward_propagation(parameters, cache, X, Y):
    """
    Implements the backward propagation, calculating gradients
    
    Arguments:
    parameters -- python dictionary containing our parameters 
    cache -- python dictionary containing Z1, A1, Z2, A2
    X -- input data of shape (n_x, number of examples)
    Y -- "true" labels vector of shape (n_y, number of examples)
    
    Returns:
    grads -- python dictionary containing gradients with respect to different parameters
    """
    m = X.shape[1]
    
    # First, retrieve W from the dictionary "parameters".
    W1 = parameters["W1"]
    W2 = parameters["W2"]
    
    # Retrieve also A1 and A2 from dictionary "cache".
    A1 = cache["A1"]
    A2 = cache["A2"]
    
    # Backward propagation: calculate partial derivatives denoted as dW1, db1, dW2, db2 for simplicity. 
    dZ2 = A2 - Y
    dW2 = 1/m * np.matmul(dZ2, A1.T)
    db2 = 1/m * np.sum(dZ2, axis = 1, keepdims = True)
    dZ1 = np.dot(W2.T, dZ2) * A1 * (1 - A1)
    dW1 = 1/m * np.matmul(dZ1, X.T)
    db1 = 1/m * np.sum(dZ1, axis = 1, keepdims = True)
    
    grads = {"dW1": dW1,
             "db1": db1,
             "dW2": dW2,
             "db2": db2}
    
    return grads

grads = backward_propagation(parameters, cache, X, Y)

print("dW1 = " + str(grads["dW1"]))
print("db1 = " + str(grads["db1"]))
print("dW2 = " + str(grads["dW2"]))
print("db2 = " + str(grads["db2"]))
```

    dW1 = [[0.14210987 0.18439398]
     [0.54176591 0.71127689]]
    db1 = [[0.03282164]
     [0.12587721]]
    dW2 = [[-0.04103538 -0.0370702 ]]
    db2 = [[-0.07808718]]


<a name='ex06'></a>
### Exercise 6

Implement `update_parameters()`.

**Instructions**:
- Update parameters as shown in $(9)$ (section [2.3](#2.3)):
\begin{align}
W^{[1]} &= W^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[1]} },\\
b^{[1]} &= b^{[1]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[1]} },\\
W^{[2]} &= W^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial W^{[2]} },\\
b^{[2]} &= b^{[2]} - \alpha \frac{\partial \mathcal{L} }{ \partial b^{[2]} }.\\
\end{align}
- The steps you have to implement are:
    1. Retrieve each parameter from the dictionary "parameters" (which is the output of `initialize_parameters()`) by using `parameters[".."]`.
    2. Retrieve each derivative from the dictionary "grads" (which is the output of `backward_propagation()`) by using `grads[".."]`.
    3. Update parameters.


```python
def update_parameters(parameters, grads, learning_rate=1.2):
    """
    Updates parameters using the gradient descent update rule
    
    Arguments:
    parameters -- python dictionary containing parameters 
    grads -- python dictionary containing gradients
    learning_rate -- learning rate for gradient descent
    
    Returns:
    parameters -- python dictionary containing updated parameters 
    """
    # Retrieve each parameter from the dictionary "parameters".
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = parameters["W1"]
    b1 = parameters["b1"]
    W2 = parameters["W2"]
    b2 = parameters["b2"]
    ### END CODE HERE ###
    
    # Retrieve each gradient from the dictionary "grads".
    ### START CODE HERE ### (~ 4 lines of code)
    dW1 = grads["dW1"]
    db1 = grads["db1"]
    dW2 = grads["dW2"]
    db2 = grads["db2"]
    ### END CODE HERE ###
    
    # Update rule for each parameter.
    ### START CODE HERE ### (~ 4 lines of code)
    W1 = W1 - learning_rate * dW1
    b1 = b1 - learning_rate * db1
    W2 = W2 - learning_rate * dW2
    b2 = b2 - learning_rate * db2
    ### END CODE HERE ###
    
    parameters = {"W1": W1,
                  "b1": b1,
                  "W2": W2,
                  "b2": b2}
    
    return parameters
```


```python
parameters_updated = update_parameters(parameters, grads)

print("W1 updated = " + str(parameters_updated["W1"]))
print("b1 updated = " + str(parameters_updated["b1"]))
print("W2 updated = " + str(parameters_updated["W2"]))
print("b2 updated = " + str(parameters_updated["b2"]))
```

    W1 updated = [[-2.05570928 -2.23173823]
     [ 1.58318226 -2.74051029]]
    b1 updated = [[-0.96248367]
     [-8.75600619]]
    W2 updated = [[-1.63880993 -6.42146224]]
    b2 updated = [[0.68870805]]


##### __Expected Output__ 
Note: the actual values can be different!

```Python
W1 updated = [[ 0.01790427  0.00434496]
 [ 0.00099046 -0.01866419]]
b1 updated = [[-6.13449205e-07]
 [-8.47483463e-07]]
W2 updated = [[-0.00238219 -0.00323487]]
b2 updated = [[0.00094478]]
```


```python
w3_unittest.test_update_parameters(update_parameters)
```

    [92m All tests passed


<a name='3.4'></a>
### 3.4 - Integrate parts 3.1, 3.2 and 3.3 in nn_model()

<a name='ex07'></a>
### Exercise 7

Build your neural network model in `nn_model()`.

**Instructions**: The neural network model has to use the previous functions in the right order.


```python
# GRADED FUNCTION: nn_model

def nn_model(X, Y, n_h, num_iterations=10, learning_rate=1.2, print_cost=False):
    """
    Arguments:
    X -- dataset of shape (n_x, number of examples)
    Y -- labels of shape (n_y, number of examples)
    num_iterations -- number of iterations in the loop
    learning_rate -- learning rate parameter for gradient descent
    print_cost -- if True, print the cost every iteration
    
    Returns:
    parameters -- parameters learnt by the model. They can then be used to predict.
    """
    
    n_x = layer_sizes(X, Y)[0]
    n_y = layer_sizes(X, Y)[2]
    
    # Initialize parameters.
    ### START CODE HERE ### (~ 1 line of code)
    parameters = initialize_parameters(n_x, n_h, n_y)
    ### END CODE HERE ###
    
    # Loop.
    for i in range(0, num_iterations):
         
        ### START CODE HERE ### (~ 4 lines of code)
        # Forward propagation. Inputs: "X, parameters". Outputs: "A2, cache".
        A2, cache = forward_propagation(X, parameters)
        
        # Cost function. Inputs: "A2, Y". Outputs: "cost".
        cost = compute_cost(A2, Y)
        
        # Backpropagation. Inputs: "parameters, cache, X, Y". Outputs: "grads".
        grads = backward_propagation(parameters, cache, X, Y)
        
        # Gradient descent parameter update. Inputs: "parameters, grads, learning_rate". Outputs: "parameters".
        parameters = update_parameters(parameters, grads, learning_rate)
        ### END CODE HERE ###
        
        # Print the cost every iteration.
        if print_cost:
            print ("Cost after iteration %i: %f" %(i, cost))

    return parameters
```


```python
parameters = nn_model(X, Y, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=True)
print("W1 = " + str(parameters["W1"]))
print("b1 = " + str(parameters["b1"]))
print("W2 = " + str(parameters["W2"]))
print("b2 = " + str(parameters["b2"]))

W1 = parameters["W1"]
b1 = parameters["b1"]
W2 = parameters["W2"]
b2 = parameters["b2"]
```

    Cost after iteration 0: 0.705432
    Cost after iteration 1: 0.695141
    Cost after iteration 2: 0.693800
    Cost after iteration 3: 0.693455
    Cost after iteration 4: 0.693317
    Cost after iteration 5: 0.693236
    Cost after iteration 6: 0.693175
    Cost after iteration 7: 0.693125
    Cost after iteration 8: 0.693080
    Cost after iteration 9: 0.693040
    Cost after iteration 10: 0.693004
    Cost after iteration 11: 0.692969
    Cost after iteration 12: 0.692936
    Cost after iteration 13: 0.692904
    Cost after iteration 14: 0.692873
    Cost after iteration 15: 0.692841
    Cost after iteration 16: 0.692808
    Cost after iteration 17: 0.692773
    Cost after iteration 18: 0.692736
    Cost after iteration 19: 0.692695
    Cost after iteration 20: 0.692650
    Cost after iteration 21: 0.692600
    Cost after iteration 22: 0.692543
    Cost after iteration 23: 0.692477
    Cost after iteration 24: 0.692402
    Cost after iteration 25: 0.692314
    Cost after iteration 26: 0.692210
    Cost after iteration 27: 0.692088
    Cost after iteration 28: 0.691942
    Cost after iteration 29: 0.691769
    Cost after iteration 30: 0.691563
    Cost after iteration 31: 0.691316
    Cost after iteration 32: 0.691020
    Cost after iteration 33: 0.690667
    Cost after iteration 34: 0.690246
    Cost after iteration 35: 0.689747
    Cost after iteration 36: 0.689156
    Cost after iteration 37: 0.688462
    Cost after iteration 38: 0.687652
    Cost after iteration 39: 0.686714
    Cost after iteration 40: 0.685635
    Cost after iteration 41: 0.684407
    Cost after iteration 42: 0.683022
    Cost after iteration 43: 0.681476
    Cost after iteration 44: 0.679766
    Cost after iteration 45: 0.677897
    Cost after iteration 46: 0.675874
    Cost after iteration 47: 0.673708
    Cost after iteration 48: 0.671412
    Cost after iteration 49: 0.669002
    Cost after iteration 50: 0.666496
    Cost after iteration 51: 0.663911
    Cost after iteration 52: 0.661267
    Cost after iteration 53: 0.658582
    Cost after iteration 54: 0.655874
    Cost after iteration 55: 0.653159
    Cost after iteration 56: 0.650452
    Cost after iteration 57: 0.647767
    Cost after iteration 58: 0.645115
    Cost after iteration 59: 0.642507
    Cost after iteration 60: 0.639951
    Cost after iteration 61: 0.637454
    Cost after iteration 62: 0.635022
    Cost after iteration 63: 0.632659
    Cost after iteration 64: 0.630370
    Cost after iteration 65: 0.628155
    Cost after iteration 66: 0.626017
    Cost after iteration 67: 0.623956
    Cost after iteration 68: 0.621973
    Cost after iteration 69: 0.620065
    Cost after iteration 70: 0.618232
    Cost after iteration 71: 0.616472
    Cost after iteration 72: 0.614784
    Cost after iteration 73: 0.613164
    Cost after iteration 74: 0.611611
    Cost after iteration 75: 0.610122
    Cost after iteration 76: 0.608695
    Cost after iteration 77: 0.607326
    Cost after iteration 78: 0.606013
    Cost after iteration 79: 0.604754
    Cost after iteration 80: 0.603546
    Cost after iteration 81: 0.602386
    Cost after iteration 82: 0.601273
    Cost after iteration 83: 0.600204
    Cost after iteration 84: 0.599176
    Cost after iteration 85: 0.598188
    Cost after iteration 86: 0.597238
    Cost after iteration 87: 0.596324
    Cost after iteration 88: 0.595444
    Cost after iteration 89: 0.594596
    Cost after iteration 90: 0.593779
    Cost after iteration 91: 0.592991
    Cost after iteration 92: 0.592231
    Cost after iteration 93: 0.591497
    Cost after iteration 94: 0.590788
    Cost after iteration 95: 0.590104
    Cost after iteration 96: 0.589442
    Cost after iteration 97: 0.588802
    Cost after iteration 98: 0.588182
    Cost after iteration 99: 0.587582
    Cost after iteration 100: 0.587001
    Cost after iteration 101: 0.586438
    Cost after iteration 102: 0.585892
    Cost after iteration 103: 0.585362
    Cost after iteration 104: 0.584847
    Cost after iteration 105: 0.584347
    Cost after iteration 106: 0.583862
    Cost after iteration 107: 0.583390
    Cost after iteration 108: 0.582930
    Cost after iteration 109: 0.582484
    Cost after iteration 110: 0.582048
    Cost after iteration 111: 0.581624
    Cost after iteration 112: 0.581211
    Cost after iteration 113: 0.580808
    Cost after iteration 114: 0.580415
    Cost after iteration 115: 0.580032
    Cost after iteration 116: 0.579657
    Cost after iteration 117: 0.579292
    Cost after iteration 118: 0.578934
    Cost after iteration 119: 0.578584
    Cost after iteration 120: 0.578242
    Cost after iteration 121: 0.577908
    Cost after iteration 122: 0.577580
    Cost after iteration 123: 0.577259
    Cost after iteration 124: 0.576944
    Cost after iteration 125: 0.576636
    Cost after iteration 126: 0.576333
    Cost after iteration 127: 0.576036
    Cost after iteration 128: 0.575745
    Cost after iteration 129: 0.575458
    Cost after iteration 130: 0.575176
    Cost after iteration 131: 0.574898
    Cost after iteration 132: 0.574625
    Cost after iteration 133: 0.574355
    Cost after iteration 134: 0.574088
    Cost after iteration 135: 0.573824
    Cost after iteration 136: 0.573562
    Cost after iteration 137: 0.573302
    Cost after iteration 138: 0.573043
    Cost after iteration 139: 0.572783
    Cost after iteration 140: 0.572522
    Cost after iteration 141: 0.572258
    Cost after iteration 142: 0.571990
    Cost after iteration 143: 0.571715
    Cost after iteration 144: 0.571431
    Cost after iteration 145: 0.571136
    Cost after iteration 146: 0.570825
    Cost after iteration 147: 0.570494
    Cost after iteration 148: 0.570140
    Cost after iteration 149: 0.569756
    Cost after iteration 150: 0.569338
    Cost after iteration 151: 0.568877
    Cost after iteration 152: 0.568369
    Cost after iteration 153: 0.567803
    Cost after iteration 154: 0.567172
    Cost after iteration 155: 0.566464
    Cost after iteration 156: 0.565664
    Cost after iteration 157: 0.564753
    Cost after iteration 158: 0.563703
    Cost after iteration 159: 0.562476
    Cost after iteration 160: 0.561023
    Cost after iteration 161: 0.559285
    Cost after iteration 162: 0.557200
    Cost after iteration 163: 0.554717
    Cost after iteration 164: 0.551801
    Cost after iteration 165: 0.548429
    Cost after iteration 166: 0.544588
    Cost after iteration 167: 0.540277
    Cost after iteration 168: 0.535514
    Cost after iteration 169: 0.530331
    Cost after iteration 170: 0.524778
    Cost after iteration 171: 0.518910
    Cost after iteration 172: 0.512791
    Cost after iteration 173: 0.506484
    Cost after iteration 174: 0.500047
    Cost after iteration 175: 0.493537
    Cost after iteration 176: 0.487002
    Cost after iteration 177: 0.480485
    Cost after iteration 178: 0.474020
    Cost after iteration 179: 0.467637
    Cost after iteration 180: 0.461358
    Cost after iteration 181: 0.455203
    Cost after iteration 182: 0.449185
    Cost after iteration 183: 0.443315
    Cost after iteration 184: 0.437599
    Cost after iteration 185: 0.432043
    Cost after iteration 186: 0.426648
    Cost after iteration 187: 0.421417
    Cost after iteration 188: 0.416349
    Cost after iteration 189: 0.411441
    Cost after iteration 190: 0.406692
    Cost after iteration 191: 0.402098
    Cost after iteration 192: 0.397655
    Cost after iteration 193: 0.393360
    Cost after iteration 194: 0.389208
    Cost after iteration 195: 0.385194
    Cost after iteration 196: 0.381315
    Cost after iteration 197: 0.377564
    Cost after iteration 198: 0.373938
    Cost after iteration 199: 0.370431
    Cost after iteration 200: 0.367040
    Cost after iteration 201: 0.363759
    Cost after iteration 202: 0.360584
    Cost after iteration 203: 0.357511
    Cost after iteration 204: 0.354536
    Cost after iteration 205: 0.351654
    Cost after iteration 206: 0.348862
    Cost after iteration 207: 0.346156
    Cost after iteration 208: 0.343532
    Cost after iteration 209: 0.340988
    Cost after iteration 210: 0.338519
    Cost after iteration 211: 0.336122
    Cost after iteration 212: 0.333796
    Cost after iteration 213: 0.331535
    Cost after iteration 214: 0.329339
    Cost after iteration 215: 0.327204
    Cost after iteration 216: 0.325127
    Cost after iteration 217: 0.323107
    Cost after iteration 218: 0.321141
    Cost after iteration 219: 0.319226
    Cost after iteration 220: 0.317362
    Cost after iteration 221: 0.315545
    Cost after iteration 222: 0.313773
    Cost after iteration 223: 0.312046
    Cost after iteration 224: 0.310361
    Cost after iteration 225: 0.308717
    Cost after iteration 226: 0.307112
    Cost after iteration 227: 0.305544
    Cost after iteration 228: 0.304012
    Cost after iteration 229: 0.302515
    Cost after iteration 230: 0.301052
    Cost after iteration 231: 0.299620
    Cost after iteration 232: 0.298220
    Cost after iteration 233: 0.296849
    Cost after iteration 234: 0.295507
    Cost after iteration 235: 0.294193
    Cost after iteration 236: 0.292905
    Cost after iteration 237: 0.291643
    Cost after iteration 238: 0.290406
    Cost after iteration 239: 0.289193
    Cost after iteration 240: 0.288002
    Cost after iteration 241: 0.286834
    Cost after iteration 242: 0.285688
    Cost after iteration 243: 0.284562
    Cost after iteration 244: 0.283457
    Cost after iteration 245: 0.282370
    Cost after iteration 246: 0.281303
    Cost after iteration 247: 0.280253
    Cost after iteration 248: 0.279221
    Cost after iteration 249: 0.278206
    Cost after iteration 250: 0.277207
    Cost after iteration 251: 0.276224
    Cost after iteration 252: 0.275256
    Cost after iteration 253: 0.274303
    Cost after iteration 254: 0.273365
    Cost after iteration 255: 0.272440
    Cost after iteration 256: 0.271529
    Cost after iteration 257: 0.270631
    Cost after iteration 258: 0.269745
    Cost after iteration 259: 0.268872
    Cost after iteration 260: 0.268011
    Cost after iteration 261: 0.267161
    Cost after iteration 262: 0.266322
    Cost after iteration 263: 0.265494
    Cost after iteration 264: 0.264677
    Cost after iteration 265: 0.263870
    Cost after iteration 266: 0.263073
    Cost after iteration 267: 0.262286
    Cost after iteration 268: 0.261508
    Cost after iteration 269: 0.260740
    Cost after iteration 270: 0.259980
    Cost after iteration 271: 0.259229
    Cost after iteration 272: 0.258486
    Cost after iteration 273: 0.257752
    Cost after iteration 274: 0.257025
    Cost after iteration 275: 0.256306
    Cost after iteration 276: 0.255595
    Cost after iteration 277: 0.254892
    Cost after iteration 278: 0.254195
    Cost after iteration 279: 0.253506
    Cost after iteration 280: 0.252823
    Cost after iteration 281: 0.252147
    Cost after iteration 282: 0.251478
    Cost after iteration 283: 0.250816
    Cost after iteration 284: 0.250159
    Cost after iteration 285: 0.249509
    Cost after iteration 286: 0.248864
    Cost after iteration 287: 0.248226
    Cost after iteration 288: 0.247594
    Cost after iteration 289: 0.246967
    Cost after iteration 290: 0.246345
    Cost after iteration 291: 0.245730
    Cost after iteration 292: 0.245119
    Cost after iteration 293: 0.244514
    Cost after iteration 294: 0.243914
    Cost after iteration 295: 0.243319
    Cost after iteration 296: 0.242729
    Cost after iteration 297: 0.242144
    Cost after iteration 298: 0.241564
    Cost after iteration 299: 0.240989
    Cost after iteration 300: 0.240418
    Cost after iteration 301: 0.239853
    Cost after iteration 302: 0.239291
    Cost after iteration 303: 0.238735
    Cost after iteration 304: 0.238183
    Cost after iteration 305: 0.237635
    Cost after iteration 306: 0.237092
    Cost after iteration 307: 0.236553
    Cost after iteration 308: 0.236019
    Cost after iteration 309: 0.235489
    Cost after iteration 310: 0.234963
    Cost after iteration 311: 0.234441
    Cost after iteration 312: 0.233924
    Cost after iteration 313: 0.233411
    Cost after iteration 314: 0.232902
    Cost after iteration 315: 0.232397
    Cost after iteration 316: 0.231896
    Cost after iteration 317: 0.231399
    Cost after iteration 318: 0.230907
    Cost after iteration 319: 0.230418
    Cost after iteration 320: 0.229934
    Cost after iteration 321: 0.229453
    Cost after iteration 322: 0.228976
    Cost after iteration 323: 0.228504
    Cost after iteration 324: 0.228035
    Cost after iteration 325: 0.227571
    Cost after iteration 326: 0.227110
    Cost after iteration 327: 0.226653
    Cost after iteration 328: 0.226200
    Cost after iteration 329: 0.225751
    Cost after iteration 330: 0.225306
    Cost after iteration 331: 0.224865
    Cost after iteration 332: 0.224428
    Cost after iteration 333: 0.223995
    Cost after iteration 334: 0.223565
    Cost after iteration 335: 0.223139
    Cost after iteration 336: 0.222718
    Cost after iteration 337: 0.222300
    Cost after iteration 338: 0.221885
    Cost after iteration 339: 0.221475
    Cost after iteration 340: 0.221068
    Cost after iteration 341: 0.220666
    Cost after iteration 342: 0.220267
    Cost after iteration 343: 0.219871
    Cost after iteration 344: 0.219480
    Cost after iteration 345: 0.219092
    Cost after iteration 346: 0.218708
    Cost after iteration 347: 0.218328
    Cost after iteration 348: 0.217951
    Cost after iteration 349: 0.217578
    Cost after iteration 350: 0.217209
    Cost after iteration 351: 0.216843
    Cost after iteration 352: 0.216481
    Cost after iteration 353: 0.216123
    Cost after iteration 354: 0.215768
    Cost after iteration 355: 0.215417
    Cost after iteration 356: 0.215069
    Cost after iteration 357: 0.214725
    Cost after iteration 358: 0.214384
    Cost after iteration 359: 0.214047
    Cost after iteration 360: 0.213714
    Cost after iteration 361: 0.213383
    Cost after iteration 362: 0.213057
    Cost after iteration 363: 0.212733
    Cost after iteration 364: 0.212413
    Cost after iteration 365: 0.212096
    Cost after iteration 366: 0.211783
    Cost after iteration 367: 0.211473
    Cost after iteration 368: 0.211166
    Cost after iteration 369: 0.210862
    Cost after iteration 370: 0.210562
    Cost after iteration 371: 0.210264
    Cost after iteration 372: 0.209970
    Cost after iteration 373: 0.209679
    Cost after iteration 374: 0.209392
    Cost after iteration 375: 0.209107
    Cost after iteration 376: 0.208825
    Cost after iteration 377: 0.208546
    Cost after iteration 378: 0.208271
    Cost after iteration 379: 0.208002
    Cost after iteration 380: 0.207744
    Cost after iteration 381: 0.207530
    Cost after iteration 382: 0.207491
    Cost after iteration 383: 0.208200
    Cost after iteration 384: 0.212198
    Cost after iteration 385: 0.231524
    Cost after iteration 386: 0.330088
    Cost after iteration 387: 0.716968
    Cost after iteration 388: 0.409842
    Cost after iteration 389: 0.660126
    Cost after iteration 390: 0.477397
    Cost after iteration 391: 0.754542
    Cost after iteration 392: 0.306270
    Cost after iteration 393: 0.254557
    Cost after iteration 394: 0.233819
    Cost after iteration 395: 0.225972
    Cost after iteration 396: 0.223169
    Cost after iteration 397: 0.221535
    Cost after iteration 398: 0.220293
    Cost after iteration 399: 0.219283
    Cost after iteration 400: 0.218391
    Cost after iteration 401: 0.217590
    Cost after iteration 402: 0.216853
    Cost after iteration 403: 0.216168
    Cost after iteration 404: 0.215525
    Cost after iteration 405: 0.214919
    Cost after iteration 406: 0.214344
    Cost after iteration 407: 0.213796
    Cost after iteration 408: 0.213271
    Cost after iteration 409: 0.212766
    Cost after iteration 410: 0.212280
    Cost after iteration 411: 0.211811
    Cost after iteration 412: 0.211355
    Cost after iteration 413: 0.210913
    Cost after iteration 414: 0.210483
    Cost after iteration 415: 0.210063
    Cost after iteration 416: 0.209654
    Cost after iteration 417: 0.209253
    Cost after iteration 418: 0.208861
    Cost after iteration 419: 0.208477
    Cost after iteration 420: 0.208100
    Cost after iteration 421: 0.207730
    Cost after iteration 422: 0.207367
    Cost after iteration 423: 0.207010
    Cost after iteration 424: 0.206660
    Cost after iteration 425: 0.206315
    Cost after iteration 426: 0.205977
    Cost after iteration 427: 0.205643
    Cost after iteration 428: 0.205316
    Cost after iteration 429: 0.204993
    Cost after iteration 430: 0.204677
    Cost after iteration 431: 0.204365
    Cost after iteration 432: 0.204058
    Cost after iteration 433: 0.203757
    Cost after iteration 434: 0.203460
    Cost after iteration 435: 0.203169
    Cost after iteration 436: 0.202883
    Cost after iteration 437: 0.202602
    Cost after iteration 438: 0.202326
    Cost after iteration 439: 0.202055
    Cost after iteration 440: 0.201790
    Cost after iteration 441: 0.201532
    Cost after iteration 442: 0.201282
    Cost after iteration 443: 0.201043
    Cost after iteration 444: 0.200822
    Cost after iteration 445: 0.200632
    Cost after iteration 446: 0.200500
    Cost after iteration 447: 0.200487
    Cost after iteration 448: 0.200726
    Cost after iteration 449: 0.201534
    Cost after iteration 450: 0.203640
    Cost after iteration 451: 0.209003
    Cost after iteration 452: 0.222884
    Cost after iteration 453: 0.262400
    Cost after iteration 454: 0.369440
    Cost after iteration 455: 0.619934
    Cost after iteration 456: 0.469353
    Cost after iteration 457: 0.667331
    Cost after iteration 458: 0.420112
    Cost after iteration 459: 0.465839
    Cost after iteration 460: 0.260782
    Cost after iteration 461: 0.237361
    Cost after iteration 462: 0.219499
    Cost after iteration 463: 0.215643
    Cost after iteration 464: 0.213218
    Cost after iteration 465: 0.211996
    Cost after iteration 466: 0.211096
    Cost after iteration 467: 0.210401
    Cost after iteration 468: 0.209791
    Cost after iteration 469: 0.209235
    Cost after iteration 470: 0.208712
    Cost after iteration 471: 0.208215
    Cost after iteration 472: 0.207737
    Cost after iteration 473: 0.207277
    Cost after iteration 474: 0.206831
    Cost after iteration 475: 0.206398
    Cost after iteration 476: 0.205978
    Cost after iteration 477: 0.205569
    Cost after iteration 478: 0.205170
    Cost after iteration 479: 0.204781
    Cost after iteration 480: 0.204402
    Cost after iteration 481: 0.204030
    Cost after iteration 482: 0.203667
    Cost after iteration 483: 0.203312
    Cost after iteration 484: 0.202964
    Cost after iteration 485: 0.202623
    Cost after iteration 486: 0.202289
    Cost after iteration 487: 0.201962
    Cost after iteration 488: 0.201641
    Cost after iteration 489: 0.201326
    Cost after iteration 490: 0.201018
    Cost after iteration 491: 0.200715
    Cost after iteration 492: 0.200418
    Cost after iteration 493: 0.200127
    Cost after iteration 494: 0.199842
    Cost after iteration 495: 0.199562
    Cost after iteration 496: 0.199287
    Cost after iteration 497: 0.199018
    Cost after iteration 498: 0.198754
    Cost after iteration 499: 0.198495
    Cost after iteration 500: 0.198242
    Cost after iteration 501: 0.197993
    Cost after iteration 502: 0.197750
    Cost after iteration 503: 0.197511
    Cost after iteration 504: 0.197277
    Cost after iteration 505: 0.197048
    Cost after iteration 506: 0.196824
    Cost after iteration 507: 0.196605
    Cost after iteration 508: 0.196390
    Cost after iteration 509: 0.196179
    Cost after iteration 510: 0.195973
    Cost after iteration 511: 0.195771
    Cost after iteration 512: 0.195574
    Cost after iteration 513: 0.195381
    Cost after iteration 514: 0.195192
    Cost after iteration 515: 0.195007
    Cost after iteration 516: 0.194826
    Cost after iteration 517: 0.194649
    Cost after iteration 518: 0.194477
    Cost after iteration 519: 0.194311
    Cost after iteration 520: 0.194151
    Cost after iteration 521: 0.194001
    Cost after iteration 522: 0.193869
    Cost after iteration 523: 0.193775
    Cost after iteration 524: 0.193760
    Cost after iteration 525: 0.193931
    Cost after iteration 526: 0.194533
    Cost after iteration 527: 0.196239
    Cost after iteration 528: 0.200624
    Cost after iteration 529: 0.213020
    Cost after iteration 530: 0.248224
    Cost after iteration 531: 0.371546
    Cost after iteration 532: 0.579668
    Cost after iteration 533: 0.846012
    Cost after iteration 534: 0.338591
    Cost after iteration 535: 0.487898
    Cost after iteration 536: 0.256424
    Cost after iteration 537: 0.250523
    Cost after iteration 538: 0.214122
    Cost after iteration 539: 0.210064
    Cost after iteration 540: 0.206443
    Cost after iteration 541: 0.204832
    Cost after iteration 542: 0.203631
    Cost after iteration 543: 0.202833
    Cost after iteration 544: 0.202176
    Cost after iteration 545: 0.201624
    Cost after iteration 546: 0.201127
    Cost after iteration 547: 0.200670
    Cost after iteration 548: 0.200241
    Cost after iteration 549: 0.199834
    Cost after iteration 550: 0.199447
    Cost after iteration 551: 0.199076
    Cost after iteration 552: 0.198719
    Cost after iteration 553: 0.198377
    Cost after iteration 554: 0.198047
    Cost after iteration 555: 0.197728
    Cost after iteration 556: 0.197421
    Cost after iteration 557: 0.197123
    Cost after iteration 558: 0.196835
    Cost after iteration 559: 0.196557
    Cost after iteration 560: 0.196287
    Cost after iteration 561: 0.196025
    Cost after iteration 562: 0.195771
    Cost after iteration 563: 0.195525
    Cost after iteration 564: 0.195285
    Cost after iteration 565: 0.195053
    Cost after iteration 566: 0.194827
    Cost after iteration 567: 0.194607
    Cost after iteration 568: 0.194394
    Cost after iteration 569: 0.194186
    Cost after iteration 570: 0.193985
    Cost after iteration 571: 0.193788
    Cost after iteration 572: 0.193597
    Cost after iteration 573: 0.193411
    Cost after iteration 574: 0.193230
    Cost after iteration 575: 0.193054
    Cost after iteration 576: 0.192883
    Cost after iteration 577: 0.192716
    Cost after iteration 578: 0.192554
    Cost after iteration 579: 0.192398
    Cost after iteration 580: 0.192247
    Cost after iteration 581: 0.192102
    Cost after iteration 582: 0.191966
    Cost after iteration 583: 0.191842
    Cost after iteration 584: 0.191738
    Cost after iteration 585: 0.191669
    Cost after iteration 586: 0.191661
    Cost after iteration 587: 0.191779
    Cost after iteration 588: 0.192131
    Cost after iteration 589: 0.193011
    Cost after iteration 590: 0.194900
    Cost after iteration 591: 0.199370
    Cost after iteration 592: 0.209113
    Cost after iteration 593: 0.235791
    Cost after iteration 594: 0.300342
    Cost after iteration 595: 0.480442
    Cost after iteration 596: 0.518872
    Cost after iteration 597: 0.632720
    Cost after iteration 598: 0.346112
    Cost after iteration 599: 0.405272
    Cost after iteration 600: 0.271856
    Cost after iteration 601: 0.275468
    Cost after iteration 602: 0.219736
    Cost after iteration 603: 0.216190
    Cost after iteration 604: 0.207827
    Cost after iteration 605: 0.205520
    Cost after iteration 606: 0.202887
    Cost after iteration 607: 0.201643
    Cost after iteration 608: 0.200551
    Cost after iteration 609: 0.199846
    Cost after iteration 610: 0.199239
    Cost after iteration 611: 0.198754
    Cost after iteration 612: 0.198315
    Cost after iteration 613: 0.197924
    Cost after iteration 614: 0.197557
    Cost after iteration 615: 0.197213
    Cost after iteration 616: 0.196884
    Cost after iteration 617: 0.196570
    Cost after iteration 618: 0.196268
    Cost after iteration 619: 0.195976
    Cost after iteration 620: 0.195695
    Cost after iteration 621: 0.195422
    Cost after iteration 622: 0.195158
    Cost after iteration 623: 0.194901
    Cost after iteration 624: 0.194653
    Cost after iteration 625: 0.194411
    Cost after iteration 626: 0.194177
    Cost after iteration 627: 0.193949
    Cost after iteration 628: 0.193727
    Cost after iteration 629: 0.193512
    Cost after iteration 630: 0.193303
    Cost after iteration 631: 0.193099
    Cost after iteration 632: 0.192902
    Cost after iteration 633: 0.192709
    Cost after iteration 634: 0.192522
    Cost after iteration 635: 0.192340
    Cost after iteration 636: 0.192163
    Cost after iteration 637: 0.191991
    Cost after iteration 638: 0.191823
    Cost after iteration 639: 0.191661
    Cost after iteration 640: 0.191502
    Cost after iteration 641: 0.191349
    Cost after iteration 642: 0.191199
    Cost after iteration 643: 0.191055
    Cost after iteration 644: 0.190914
    Cost after iteration 645: 0.190779
    Cost after iteration 646: 0.190648
    Cost after iteration 647: 0.190524
    Cost after iteration 648: 0.190405
    Cost after iteration 649: 0.190295
    Cost after iteration 650: 0.190194
    Cost after iteration 651: 0.190108
    Cost after iteration 652: 0.190041
    Cost after iteration 653: 0.190009
    Cost after iteration 654: 0.190019
    Cost after iteration 655: 0.190123
    Cost after iteration 656: 0.190336
    Cost after iteration 657: 0.190834
    Cost after iteration 658: 0.191640
    Cost after iteration 659: 0.193408
    Cost after iteration 660: 0.196128
    Cost after iteration 661: 0.202533
    Cost after iteration 662: 0.212315
    Cost after iteration 663: 0.238920
    Cost after iteration 664: 0.280410
    Cost after iteration 665: 0.426870
    Cost after iteration 666: 0.552449
    Cost after iteration 667: 1.130558
    Cost after iteration 668: 0.277537
    Cost after iteration 669: 0.258112
    Cost after iteration 670: 0.208139
    Cost after iteration 671: 0.204132
    Cost after iteration 672: 0.202117
    Cost after iteration 673: 0.201076
    Cost after iteration 674: 0.200331
    Cost after iteration 675: 0.199720
    Cost after iteration 676: 0.199172
    Cost after iteration 677: 0.198665
    Cost after iteration 678: 0.198189
    Cost after iteration 679: 0.197739
    Cost after iteration 680: 0.197313
    Cost after iteration 681: 0.196907
    Cost after iteration 682: 0.196522
    Cost after iteration 683: 0.196154
    Cost after iteration 684: 0.195804
    Cost after iteration 685: 0.195469
    Cost after iteration 686: 0.195150
    Cost after iteration 687: 0.194844
    Cost after iteration 688: 0.194552
    Cost after iteration 689: 0.194272
    Cost after iteration 690: 0.194003
    Cost after iteration 691: 0.193745
    Cost after iteration 692: 0.193498
    Cost after iteration 693: 0.193260
    Cost after iteration 694: 0.193031
    Cost after iteration 695: 0.192811
    Cost after iteration 696: 0.192599
    Cost after iteration 697: 0.192395
    Cost after iteration 698: 0.192198
    Cost after iteration 699: 0.192008
    Cost after iteration 700: 0.191825
    Cost after iteration 701: 0.191648
    Cost after iteration 702: 0.191478
    Cost after iteration 703: 0.191313
    Cost after iteration 704: 0.191153
    Cost after iteration 705: 0.190998
    Cost after iteration 706: 0.190849
    Cost after iteration 707: 0.190704
    Cost after iteration 708: 0.190564
    Cost after iteration 709: 0.190428
    Cost after iteration 710: 0.190297
    Cost after iteration 711: 0.190169
    Cost after iteration 712: 0.190045
    Cost after iteration 713: 0.189925
    Cost after iteration 714: 0.189808
    Cost after iteration 715: 0.189695
    Cost after iteration 716: 0.189585
    Cost after iteration 717: 0.189478
    Cost after iteration 718: 0.189374
    Cost after iteration 719: 0.189273
    Cost after iteration 720: 0.189175
    Cost after iteration 721: 0.189079
    Cost after iteration 722: 0.188986
    Cost after iteration 723: 0.188895
    Cost after iteration 724: 0.188807
    Cost after iteration 725: 0.188721
    Cost after iteration 726: 0.188637
    Cost after iteration 727: 0.188555
    Cost after iteration 728: 0.188475
    Cost after iteration 729: 0.188398
    Cost after iteration 730: 0.188322
    Cost after iteration 731: 0.188247
    Cost after iteration 732: 0.188175
    Cost after iteration 733: 0.188104
    Cost after iteration 734: 0.188035
    Cost after iteration 735: 0.187968
    Cost after iteration 736: 0.187902
    Cost after iteration 737: 0.187838
    Cost after iteration 738: 0.187775
    Cost after iteration 739: 0.187714
    Cost after iteration 740: 0.187656
    Cost after iteration 741: 0.187600
    Cost after iteration 742: 0.187550
    Cost after iteration 743: 0.187508
    Cost after iteration 744: 0.187484
    Cost after iteration 745: 0.187494
    Cost after iteration 746: 0.187573
    Cost after iteration 747: 0.187808
    Cost after iteration 748: 0.188353
    Cost after iteration 749: 0.189649
    Cost after iteration 750: 0.192425
    Cost after iteration 751: 0.199385
    Cost after iteration 752: 0.214858
    Cost after iteration 753: 0.264897
    Cost after iteration 754: 0.371878
    Cost after iteration 755: 0.603037
    Cost after iteration 756: 0.415302
    Cost after iteration 757: 0.542540
    Cost after iteration 758: 0.403876
    Cost after iteration 759: 0.519112
    Cost after iteration 760: 0.277047
    Cost after iteration 761: 0.255222
    Cost after iteration 762: 0.211069
    Cost after iteration 763: 0.205140
    Cost after iteration 764: 0.200848
    Cost after iteration 765: 0.199125
    Cost after iteration 766: 0.197955
    Cost after iteration 767: 0.197239
    Cost after iteration 768: 0.196681
    Cost after iteration 769: 0.196224
    Cost after iteration 770: 0.195817
    Cost after iteration 771: 0.195443
    Cost after iteration 772: 0.195093
    Cost after iteration 773: 0.194762
    Cost after iteration 774: 0.194447
    Cost after iteration 775: 0.194145
    Cost after iteration 776: 0.193857
    Cost after iteration 777: 0.193580
    Cost after iteration 778: 0.193315
    Cost after iteration 779: 0.193060
    Cost after iteration 780: 0.192815
    Cost after iteration 781: 0.192580
    Cost after iteration 782: 0.192354
    Cost after iteration 783: 0.192136
    Cost after iteration 784: 0.191926
    Cost after iteration 785: 0.191724
    Cost after iteration 786: 0.191529
    Cost after iteration 787: 0.191341
    Cost after iteration 788: 0.191160
    Cost after iteration 789: 0.190985
    Cost after iteration 790: 0.190817
    Cost after iteration 791: 0.190654
    Cost after iteration 792: 0.190497
    Cost after iteration 793: 0.190345
    Cost after iteration 794: 0.190198
    Cost after iteration 795: 0.190056
    Cost after iteration 796: 0.189918
    Cost after iteration 797: 0.189786
    Cost after iteration 798: 0.189657
    Cost after iteration 799: 0.189533
    Cost after iteration 800: 0.189412
    Cost after iteration 801: 0.189295
    Cost after iteration 802: 0.189182
    Cost after iteration 803: 0.189073
    Cost after iteration 804: 0.188967
    Cost after iteration 805: 0.188864
    Cost after iteration 806: 0.188764
    Cost after iteration 807: 0.188667
    Cost after iteration 808: 0.188573
    Cost after iteration 809: 0.188482
    Cost after iteration 810: 0.188393
    Cost after iteration 811: 0.188308
    Cost after iteration 812: 0.188224
    Cost after iteration 813: 0.188143
    Cost after iteration 814: 0.188064
    Cost after iteration 815: 0.187987
    Cost after iteration 816: 0.187913
    Cost after iteration 817: 0.187840
    Cost after iteration 818: 0.187770
    Cost after iteration 819: 0.187701
    Cost after iteration 820: 0.187634
    Cost after iteration 821: 0.187569
    Cost after iteration 822: 0.187505
    Cost after iteration 823: 0.187444
    Cost after iteration 824: 0.187384
    Cost after iteration 825: 0.187325
    Cost after iteration 826: 0.187269
    Cost after iteration 827: 0.187214
    Cost after iteration 828: 0.187161
    Cost after iteration 829: 0.187111
    Cost after iteration 830: 0.187064
    Cost after iteration 831: 0.187022
    Cost after iteration 832: 0.186987
    Cost after iteration 833: 0.186965
    Cost after iteration 834: 0.186963
    Cost after iteration 835: 0.186999
    Cost after iteration 836: 0.187094
    Cost after iteration 837: 0.187320
    Cost after iteration 838: 0.187735
    Cost after iteration 839: 0.188631
    Cost after iteration 840: 0.190155
    Cost after iteration 841: 0.193648
    Cost after iteration 842: 0.199341
    Cost after iteration 843: 0.214328
    Cost after iteration 844: 0.237617
    Cost after iteration 845: 0.314182
    Cost after iteration 846: 0.404806
    Cost after iteration 847: 0.838335
    Cost after iteration 848: 0.516924
    Cost after iteration 849: 0.708110
    Cost after iteration 850: 0.291061
    Cost after iteration 851: 0.242579
    Cost after iteration 852: 0.212409
    Cost after iteration 853: 0.202712
    Cost after iteration 854: 0.199862
    Cost after iteration 855: 0.198715
    Cost after iteration 856: 0.198044
    Cost after iteration 857: 0.197514
    Cost after iteration 858: 0.197040
    Cost after iteration 859: 0.196598
    Cost after iteration 860: 0.196182
    Cost after iteration 861: 0.195786
    Cost after iteration 862: 0.195411
    Cost after iteration 863: 0.195053
    Cost after iteration 864: 0.194712
    Cost after iteration 865: 0.194388
    Cost after iteration 866: 0.194078
    Cost after iteration 867: 0.193782
    Cost after iteration 868: 0.193500
    Cost after iteration 869: 0.193230
    Cost after iteration 870: 0.192972
    Cost after iteration 871: 0.192725
    Cost after iteration 872: 0.192488
    Cost after iteration 873: 0.192262
    Cost after iteration 874: 0.192044
    Cost after iteration 875: 0.191836
    Cost after iteration 876: 0.191636
    Cost after iteration 877: 0.191445
    Cost after iteration 878: 0.191260
    Cost after iteration 879: 0.191083
    Cost after iteration 880: 0.190913
    Cost after iteration 881: 0.190749
    Cost after iteration 882: 0.190591
    Cost after iteration 883: 0.190440
    Cost after iteration 884: 0.190293
    Cost after iteration 885: 0.190152
    Cost after iteration 886: 0.190016
    Cost after iteration 887: 0.189885
    Cost after iteration 888: 0.189759
    Cost after iteration 889: 0.189636
    Cost after iteration 890: 0.189518
    Cost after iteration 891: 0.189404
    Cost after iteration 892: 0.189294
    Cost after iteration 893: 0.189187
    Cost after iteration 894: 0.189084
    Cost after iteration 895: 0.188984
    Cost after iteration 896: 0.188887
    Cost after iteration 897: 0.188793
    Cost after iteration 898: 0.188702
    Cost after iteration 899: 0.188614
    Cost after iteration 900: 0.188529
    Cost after iteration 901: 0.188446
    Cost after iteration 902: 0.188365
    Cost after iteration 903: 0.188287
    Cost after iteration 904: 0.188211
    Cost after iteration 905: 0.188137
    Cost after iteration 906: 0.188065
    Cost after iteration 907: 0.187995
    Cost after iteration 908: 0.187927
    Cost after iteration 909: 0.187861
    Cost after iteration 910: 0.187797
    Cost after iteration 911: 0.187734
    Cost after iteration 912: 0.187673
    Cost after iteration 913: 0.187613
    Cost after iteration 914: 0.187555
    Cost after iteration 915: 0.187498
    Cost after iteration 916: 0.187443
    Cost after iteration 917: 0.187389
    Cost after iteration 918: 0.187336
    Cost after iteration 919: 0.187285
    Cost after iteration 920: 0.187235
    Cost after iteration 921: 0.187185
    Cost after iteration 922: 0.187137
    Cost after iteration 923: 0.187090
    Cost after iteration 924: 0.187044
    Cost after iteration 925: 0.186999
    Cost after iteration 926: 0.186955
    Cost after iteration 927: 0.186912
    Cost after iteration 928: 0.186870
    Cost after iteration 929: 0.186828
    Cost after iteration 930: 0.186788
    Cost after iteration 931: 0.186748
    Cost after iteration 932: 0.186709
    Cost after iteration 933: 0.186670
    Cost after iteration 934: 0.186633
    Cost after iteration 935: 0.186596
    Cost after iteration 936: 0.186560
    Cost after iteration 937: 0.186524
    Cost after iteration 938: 0.186489
    Cost after iteration 939: 0.186455
    Cost after iteration 940: 0.186421
    Cost after iteration 941: 0.186388
    Cost after iteration 942: 0.186355
    Cost after iteration 943: 0.186323
    Cost after iteration 944: 0.186291
    Cost after iteration 945: 0.186260
    Cost after iteration 946: 0.186229
    Cost after iteration 947: 0.186199
    Cost after iteration 948: 0.186170
    Cost after iteration 949: 0.186140
    Cost after iteration 950: 0.186112
    Cost after iteration 951: 0.186083
    Cost after iteration 952: 0.186056
    Cost after iteration 953: 0.186029
    Cost after iteration 954: 0.186004
    Cost after iteration 955: 0.185981
    Cost after iteration 956: 0.185963
    Cost after iteration 957: 0.185953
    Cost after iteration 958: 0.185959
    Cost after iteration 959: 0.185999
    Cost after iteration 960: 0.186105
    Cost after iteration 961: 0.186356
    Cost after iteration 962: 0.186884
    Cost after iteration 963: 0.188074
    Cost after iteration 964: 0.190452
    Cost after iteration 965: 0.196168
    Cost after iteration 966: 0.207376
    Cost after iteration 967: 0.239992
    Cost after iteration 968: 0.306523
    Cost after iteration 969: 0.571861
    Cost after iteration 970: 0.722236
    Cost after iteration 971: 1.730605
    Cost after iteration 972: 0.784779
    Cost after iteration 973: 0.244395
    Cost after iteration 974: 0.225918
    Cost after iteration 975: 0.222189
    Cost after iteration 976: 0.219070
    Cost after iteration 977: 0.216380
    Cost after iteration 978: 0.214033
    Cost after iteration 979: 0.211966
    Cost after iteration 980: 0.210130
    Cost after iteration 981: 0.208489
    Cost after iteration 982: 0.207012
    Cost after iteration 983: 0.205677
    Cost after iteration 984: 0.204463
    Cost after iteration 985: 0.203355
    Cost after iteration 986: 0.202341
    Cost after iteration 987: 0.201409
    Cost after iteration 988: 0.200549
    Cost after iteration 989: 0.199755
    Cost after iteration 990: 0.199019
    Cost after iteration 991: 0.198335
    Cost after iteration 992: 0.197700
    Cost after iteration 993: 0.197107
    Cost after iteration 994: 0.196554
    Cost after iteration 995: 0.196037
    Cost after iteration 996: 0.195554
    Cost after iteration 997: 0.195100
    Cost after iteration 998: 0.194675
    Cost after iteration 999: 0.194275
    Cost after iteration 1000: 0.193900
    Cost after iteration 1001: 0.193546
    Cost after iteration 1002: 0.193213
    Cost after iteration 1003: 0.192900
    Cost after iteration 1004: 0.192603
    Cost after iteration 1005: 0.192324
    Cost after iteration 1006: 0.192059
    Cost after iteration 1007: 0.191809
    Cost after iteration 1008: 0.191573
    Cost after iteration 1009: 0.191349
    Cost after iteration 1010: 0.191136
    Cost after iteration 1011: 0.190935
    Cost after iteration 1012: 0.190743
    Cost after iteration 1013: 0.190562
    Cost after iteration 1014: 0.190389
    Cost after iteration 1015: 0.190224
    Cost after iteration 1016: 0.190068
    Cost after iteration 1017: 0.189919
    Cost after iteration 1018: 0.189776
    Cost after iteration 1019: 0.189641
    Cost after iteration 1020: 0.189511
    Cost after iteration 1021: 0.189387
    Cost after iteration 1022: 0.189269
    Cost after iteration 1023: 0.189156
    Cost after iteration 1024: 0.189047
    Cost after iteration 1025: 0.188943
    Cost after iteration 1026: 0.188844
    Cost after iteration 1027: 0.188748
    Cost after iteration 1028: 0.188656
    Cost after iteration 1029: 0.188568
    Cost after iteration 1030: 0.188483
    Cost after iteration 1031: 0.188402
    Cost after iteration 1032: 0.188323
    Cost after iteration 1033: 0.188247
    Cost after iteration 1034: 0.188174
    Cost after iteration 1035: 0.188104
    Cost after iteration 1036: 0.188035
    Cost after iteration 1037: 0.187970
    Cost after iteration 1038: 0.187906
    Cost after iteration 1039: 0.187845
    Cost after iteration 1040: 0.187785
    Cost after iteration 1041: 0.187727
    Cost after iteration 1042: 0.187671
    Cost after iteration 1043: 0.187617
    Cost after iteration 1044: 0.187564
    Cost after iteration 1045: 0.187513
    Cost after iteration 1046: 0.187464
    Cost after iteration 1047: 0.187415
    Cost after iteration 1048: 0.187368
    Cost after iteration 1049: 0.187323
    Cost after iteration 1050: 0.187278
    Cost after iteration 1051: 0.187235
    Cost after iteration 1052: 0.187193
    Cost after iteration 1053: 0.187151
    Cost after iteration 1054: 0.187111
    Cost after iteration 1055: 0.187072
    Cost after iteration 1056: 0.187034
    Cost after iteration 1057: 0.186996
    Cost after iteration 1058: 0.186960
    Cost after iteration 1059: 0.186924
    Cost after iteration 1060: 0.186889
    Cost after iteration 1061: 0.186854
    Cost after iteration 1062: 0.186821
    Cost after iteration 1063: 0.186788
    Cost after iteration 1064: 0.186756
    Cost after iteration 1065: 0.186724
    Cost after iteration 1066: 0.186693
    Cost after iteration 1067: 0.186662
    Cost after iteration 1068: 0.186632
    Cost after iteration 1069: 0.186603
    Cost after iteration 1070: 0.186574
    Cost after iteration 1071: 0.186546
    Cost after iteration 1072: 0.186518
    Cost after iteration 1073: 0.186490
    Cost after iteration 1074: 0.186463
    Cost after iteration 1075: 0.186437
    Cost after iteration 1076: 0.186411
    Cost after iteration 1077: 0.186385
    Cost after iteration 1078: 0.186360
    Cost after iteration 1079: 0.186335
    Cost after iteration 1080: 0.186310
    Cost after iteration 1081: 0.186286
    Cost after iteration 1082: 0.186262
    Cost after iteration 1083: 0.186238
    Cost after iteration 1084: 0.186215
    Cost after iteration 1085: 0.186192
    Cost after iteration 1086: 0.186169
    Cost after iteration 1087: 0.186147
    Cost after iteration 1088: 0.186125
    Cost after iteration 1089: 0.186103
    Cost after iteration 1090: 0.186081
    Cost after iteration 1091: 0.186060
    Cost after iteration 1092: 0.186039
    Cost after iteration 1093: 0.186018
    Cost after iteration 1094: 0.185997
    Cost after iteration 1095: 0.185977
    Cost after iteration 1096: 0.185956
    Cost after iteration 1097: 0.185936
    Cost after iteration 1098: 0.185917
    Cost after iteration 1099: 0.185897
    Cost after iteration 1100: 0.185878
    Cost after iteration 1101: 0.185858
    Cost after iteration 1102: 0.185839
    Cost after iteration 1103: 0.185821
    Cost after iteration 1104: 0.185802
    Cost after iteration 1105: 0.185783
    Cost after iteration 1106: 0.185765
    Cost after iteration 1107: 0.185747
    Cost after iteration 1108: 0.185729
    Cost after iteration 1109: 0.185711
    Cost after iteration 1110: 0.185693
    Cost after iteration 1111: 0.185676
    Cost after iteration 1112: 0.185659
    Cost after iteration 1113: 0.185641
    Cost after iteration 1114: 0.185624
    Cost after iteration 1115: 0.185607
    Cost after iteration 1116: 0.185590
    Cost after iteration 1117: 0.185574
    Cost after iteration 1118: 0.185557
    Cost after iteration 1119: 0.185541
    Cost after iteration 1120: 0.185524
    Cost after iteration 1121: 0.185508
    Cost after iteration 1122: 0.185492
    Cost after iteration 1123: 0.185476
    Cost after iteration 1124: 0.185460
    Cost after iteration 1125: 0.185444
    Cost after iteration 1126: 0.185429
    Cost after iteration 1127: 0.185415
    Cost after iteration 1128: 0.185403
    Cost after iteration 1129: 0.185396
    Cost after iteration 1130: 0.185400
    Cost after iteration 1131: 0.185433
    Cost after iteration 1132: 0.185538
    Cost after iteration 1133: 0.185808
    Cost after iteration 1134: 0.186553
    Cost after iteration 1135: 0.188283
    Cost after iteration 1136: 0.193501
    Cost after iteration 1137: 0.204883
    Cost after iteration 1138: 0.244964
    Cost after iteration 1139: 0.297067
    Cost after iteration 1140: 0.553550
    Cost after iteration 1141: 0.425653
    Cost after iteration 1142: 0.819926
    Cost after iteration 1143: 0.526003
    Cost after iteration 1144: 0.580109
    Cost after iteration 1145: 0.229189
    Cost after iteration 1146: 0.210114
    Cost after iteration 1147: 0.201850
    Cost after iteration 1148: 0.198577
    Cost after iteration 1149: 0.197212
    Cost after iteration 1150: 0.196433
    Cost after iteration 1151: 0.195856
    Cost after iteration 1152: 0.195360
    Cost after iteration 1153: 0.194908
    Cost after iteration 1154: 0.194486
    Cost after iteration 1155: 0.194091
    Cost after iteration 1156: 0.193719
    Cost after iteration 1157: 0.193367
    Cost after iteration 1158: 0.193036
    Cost after iteration 1159: 0.192722
    Cost after iteration 1160: 0.192426
    Cost after iteration 1161: 0.192145
    Cost after iteration 1162: 0.191880
    Cost after iteration 1163: 0.191628
    Cost after iteration 1164: 0.191390
    Cost after iteration 1165: 0.191164
    Cost after iteration 1166: 0.190949
    Cost after iteration 1167: 0.190745
    Cost after iteration 1168: 0.190552
    Cost after iteration 1169: 0.190368
    Cost after iteration 1170: 0.190193
    Cost after iteration 1171: 0.190027
    Cost after iteration 1172: 0.189868
    Cost after iteration 1173: 0.189717
    Cost after iteration 1174: 0.189573
    Cost after iteration 1175: 0.189436
    Cost after iteration 1176: 0.189305
    Cost after iteration 1177: 0.189180
    Cost after iteration 1178: 0.189061
    Cost after iteration 1179: 0.188946
    Cost after iteration 1180: 0.188837
    Cost after iteration 1181: 0.188732
    Cost after iteration 1182: 0.188632
    Cost after iteration 1183: 0.188535
    Cost after iteration 1184: 0.188443
    Cost after iteration 1185: 0.188354
    Cost after iteration 1186: 0.188269
    Cost after iteration 1187: 0.188187
    Cost after iteration 1188: 0.188109
    Cost after iteration 1189: 0.188033
    Cost after iteration 1190: 0.187960
    Cost after iteration 1191: 0.187889
    Cost after iteration 1192: 0.187821
    Cost after iteration 1193: 0.187756
    Cost after iteration 1194: 0.187692
    Cost after iteration 1195: 0.187631
    Cost after iteration 1196: 0.187572
    Cost after iteration 1197: 0.187515
    Cost after iteration 1198: 0.187459
    Cost after iteration 1199: 0.187405
    Cost after iteration 1200: 0.187353
    Cost after iteration 1201: 0.187302
    Cost after iteration 1202: 0.187253
    Cost after iteration 1203: 0.187205
    Cost after iteration 1204: 0.187159
    Cost after iteration 1205: 0.187114
    Cost after iteration 1206: 0.187070
    Cost after iteration 1207: 0.187027
    Cost after iteration 1208: 0.186985
    Cost after iteration 1209: 0.186945
    Cost after iteration 1210: 0.186905
    Cost after iteration 1211: 0.186866
    Cost after iteration 1212: 0.186829
    Cost after iteration 1213: 0.186792
    Cost after iteration 1214: 0.186756
    Cost after iteration 1215: 0.186720
    Cost after iteration 1216: 0.186686
    Cost after iteration 1217: 0.186652
    Cost after iteration 1218: 0.186619
    Cost after iteration 1219: 0.186587
    Cost after iteration 1220: 0.186555
    Cost after iteration 1221: 0.186524
    Cost after iteration 1222: 0.186493
    Cost after iteration 1223: 0.186463
    Cost after iteration 1224: 0.186434
    Cost after iteration 1225: 0.186405
    Cost after iteration 1226: 0.186377
    Cost after iteration 1227: 0.186349
    Cost after iteration 1228: 0.186321
    Cost after iteration 1229: 0.186294
    Cost after iteration 1230: 0.186268
    Cost after iteration 1231: 0.186242
    Cost after iteration 1232: 0.186216
    Cost after iteration 1233: 0.186191
    Cost after iteration 1234: 0.186166
    Cost after iteration 1235: 0.186142
    Cost after iteration 1236: 0.186118
    Cost after iteration 1237: 0.186094
    Cost after iteration 1238: 0.186071
    Cost after iteration 1239: 0.186048
    Cost after iteration 1240: 0.186025
    Cost after iteration 1241: 0.186003
    Cost after iteration 1242: 0.185982
    Cost after iteration 1243: 0.185962
    Cost after iteration 1244: 0.185944
    Cost after iteration 1245: 0.185929
    Cost after iteration 1246: 0.185920
    Cost after iteration 1247: 0.185920
    Cost after iteration 1248: 0.185937
    Cost after iteration 1249: 0.185985
    Cost after iteration 1250: 0.186093
    Cost after iteration 1251: 0.186306
    Cost after iteration 1252: 0.186737
    Cost after iteration 1253: 0.187527
    Cost after iteration 1254: 0.189160
    Cost after iteration 1255: 0.192058
    Cost after iteration 1256: 0.198585
    Cost after iteration 1257: 0.209899
    Cost after iteration 1258: 0.241424
    Cost after iteration 1259: 0.297127
    Cost after iteration 1260: 0.511472
    Cost after iteration 1261: 0.616753
    Cost after iteration 1262: 1.380358
    Cost after iteration 1263: 0.212460
    Cost after iteration 1264: 0.204597
    Cost after iteration 1265: 0.201911
    Cost after iteration 1266: 0.200576
    Cost after iteration 1267: 0.199637
    Cost after iteration 1268: 0.198849
    Cost after iteration 1269: 0.198143
    Cost after iteration 1270: 0.197493
    Cost after iteration 1271: 0.196890
    Cost after iteration 1272: 0.196327
    Cost after iteration 1273: 0.195800
    Cost after iteration 1274: 0.195307
    Cost after iteration 1275: 0.194844
    Cost after iteration 1276: 0.194408
    Cost after iteration 1277: 0.193999
    Cost after iteration 1278: 0.193614
    Cost after iteration 1279: 0.193251
    Cost after iteration 1280: 0.192909
    Cost after iteration 1281: 0.192587
    Cost after iteration 1282: 0.192282
    Cost after iteration 1283: 0.191994
    Cost after iteration 1284: 0.191722
    Cost after iteration 1285: 0.191465
    Cost after iteration 1286: 0.191222
    Cost after iteration 1287: 0.190991
    Cost after iteration 1288: 0.190773
    Cost after iteration 1289: 0.190566
    Cost after iteration 1290: 0.190369
    Cost after iteration 1291: 0.190183
    Cost after iteration 1292: 0.190006
    Cost after iteration 1293: 0.189838
    Cost after iteration 1294: 0.189678
    Cost after iteration 1295: 0.189526
    Cost after iteration 1296: 0.189382
    Cost after iteration 1297: 0.189244
    Cost after iteration 1298: 0.189113
    Cost after iteration 1299: 0.188988
    Cost after iteration 1300: 0.188869
    Cost after iteration 1301: 0.188755
    Cost after iteration 1302: 0.188646
    Cost after iteration 1303: 0.188542
    Cost after iteration 1304: 0.188443
    Cost after iteration 1305: 0.188348
    Cost after iteration 1306: 0.188257
    Cost after iteration 1307: 0.188170
    Cost after iteration 1308: 0.188087
    Cost after iteration 1309: 0.188006
    Cost after iteration 1310: 0.187930
    Cost after iteration 1311: 0.187856
    Cost after iteration 1312: 0.187785
    Cost after iteration 1313: 0.187717
    Cost after iteration 1314: 0.187651
    Cost after iteration 1315: 0.187588
    Cost after iteration 1316: 0.187528
    Cost after iteration 1317: 0.187469
    Cost after iteration 1318: 0.187412
    Cost after iteration 1319: 0.187358
    Cost after iteration 1320: 0.187305
    Cost after iteration 1321: 0.187254
    Cost after iteration 1322: 0.187205
    Cost after iteration 1323: 0.187158
    Cost after iteration 1324: 0.187112
    Cost after iteration 1325: 0.187067
    Cost after iteration 1326: 0.187024
    Cost after iteration 1327: 0.186982
    Cost after iteration 1328: 0.186941
    Cost after iteration 1329: 0.186901
    Cost after iteration 1330: 0.186863
    Cost after iteration 1331: 0.186825
    Cost after iteration 1332: 0.186789
    Cost after iteration 1333: 0.186754
    Cost after iteration 1334: 0.186719
    Cost after iteration 1335: 0.186686
    Cost after iteration 1336: 0.186653
    Cost after iteration 1337: 0.186621
    Cost after iteration 1338: 0.186590
    Cost after iteration 1339: 0.186559
    Cost after iteration 1340: 0.186530
    Cost after iteration 1341: 0.186501
    Cost after iteration 1342: 0.186472
    Cost after iteration 1343: 0.186444
    Cost after iteration 1344: 0.186417
    Cost after iteration 1345: 0.186391
    Cost after iteration 1346: 0.186365
    Cost after iteration 1347: 0.186339
    Cost after iteration 1348: 0.186314
    Cost after iteration 1349: 0.186289
    Cost after iteration 1350: 0.186265
    Cost after iteration 1351: 0.186241
    Cost after iteration 1352: 0.186218
    Cost after iteration 1353: 0.186195
    Cost after iteration 1354: 0.186173
    Cost after iteration 1355: 0.186150
    Cost after iteration 1356: 0.186129
    Cost after iteration 1357: 0.186107
    Cost after iteration 1358: 0.186086
    Cost after iteration 1359: 0.186065
    Cost after iteration 1360: 0.186045
    Cost after iteration 1361: 0.186025
    Cost after iteration 1362: 0.186005
    Cost after iteration 1363: 0.185986
    Cost after iteration 1364: 0.185967
    Cost after iteration 1365: 0.185949
    Cost after iteration 1366: 0.185933
    Cost after iteration 1367: 0.185918
    Cost after iteration 1368: 0.185908
    Cost after iteration 1369: 0.185905
    Cost after iteration 1370: 0.185917
    Cost after iteration 1371: 0.185961
    Cost after iteration 1372: 0.186061
    Cost after iteration 1373: 0.186298
    Cost after iteration 1374: 0.186749
    Cost after iteration 1375: 0.187842
    Cost after iteration 1376: 0.189766
    Cost after iteration 1377: 0.195098
    Cost after iteration 1378: 0.203825
    Cost after iteration 1379: 0.232139
    Cost after iteration 1380: 0.258684
    Cost after iteration 1381: 0.388861
    Cost after iteration 1382: 0.337368
    Cost after iteration 1383: 0.514995
    Cost after iteration 1384: 0.352995
    Cost after iteration 1385: 0.524691
    Cost after iteration 1386: 0.385088
    Cost after iteration 1387: 0.427102
    Cost after iteration 1388: 0.272441
    Cost after iteration 1389: 0.247836
    Cost after iteration 1390: 0.214027
    Cost after iteration 1391: 0.201713
    Cost after iteration 1392: 0.196680
    Cost after iteration 1393: 0.194774
    Cost after iteration 1394: 0.193943
    Cost after iteration 1395: 0.193464
    Cost after iteration 1396: 0.193102
    Cost after iteration 1397: 0.192789
    Cost after iteration 1398: 0.192503
    Cost after iteration 1399: 0.192236
    Cost after iteration 1400: 0.191983
    Cost after iteration 1401: 0.191743
    Cost after iteration 1402: 0.191516
    Cost after iteration 1403: 0.191299
    Cost after iteration 1404: 0.191093
    Cost after iteration 1405: 0.190897
    Cost after iteration 1406: 0.190710
    Cost after iteration 1407: 0.190532
    Cost after iteration 1408: 0.190362
    Cost after iteration 1409: 0.190200
    Cost after iteration 1410: 0.190045
    Cost after iteration 1411: 0.189897
    Cost after iteration 1412: 0.189755
    Cost after iteration 1413: 0.189620
    Cost after iteration 1414: 0.189491
    Cost after iteration 1415: 0.189367
    Cost after iteration 1416: 0.189248
    Cost after iteration 1417: 0.189135
    Cost after iteration 1418: 0.189026
    Cost after iteration 1419: 0.188921
    Cost after iteration 1420: 0.188821
    Cost after iteration 1421: 0.188725
    Cost after iteration 1422: 0.188632
    Cost after iteration 1423: 0.188543
    Cost after iteration 1424: 0.188458
    Cost after iteration 1425: 0.188375
    Cost after iteration 1426: 0.188296
    Cost after iteration 1427: 0.188220
    Cost after iteration 1428: 0.188146
    Cost after iteration 1429: 0.188075
    Cost after iteration 1430: 0.188007
    Cost after iteration 1431: 0.187941
    Cost after iteration 1432: 0.187877
    Cost after iteration 1433: 0.187815
    Cost after iteration 1434: 0.187755
    Cost after iteration 1435: 0.187698
    Cost after iteration 1436: 0.187642
    Cost after iteration 1437: 0.187587
    Cost after iteration 1438: 0.187535
    Cost after iteration 1439: 0.187484
    Cost after iteration 1440: 0.187435
    Cost after iteration 1441: 0.187387
    Cost after iteration 1442: 0.187340
    Cost after iteration 1443: 0.187295
    Cost after iteration 1444: 0.187251
    Cost after iteration 1445: 0.187208
    Cost after iteration 1446: 0.187167
    Cost after iteration 1447: 0.187126
    Cost after iteration 1448: 0.187087
    Cost after iteration 1449: 0.187048
    Cost after iteration 1450: 0.187011
    Cost after iteration 1451: 0.186974
    Cost after iteration 1452: 0.186939
    Cost after iteration 1453: 0.186904
    Cost after iteration 1454: 0.186870
    Cost after iteration 1455: 0.186837
    Cost after iteration 1456: 0.186804
    Cost after iteration 1457: 0.186772
    Cost after iteration 1458: 0.186741
    Cost after iteration 1459: 0.186711
    Cost after iteration 1460: 0.186681
    Cost after iteration 1461: 0.186652
    Cost after iteration 1462: 0.186623
    Cost after iteration 1463: 0.186595
    Cost after iteration 1464: 0.186568
    Cost after iteration 1465: 0.186541
    Cost after iteration 1466: 0.186514
    Cost after iteration 1467: 0.186488
    Cost after iteration 1468: 0.186463
    Cost after iteration 1469: 0.186438
    Cost after iteration 1470: 0.186413
    Cost after iteration 1471: 0.186389
    Cost after iteration 1472: 0.186365
    Cost after iteration 1473: 0.186342
    Cost after iteration 1474: 0.186319
    Cost after iteration 1475: 0.186296
    Cost after iteration 1476: 0.186274
    Cost after iteration 1477: 0.186252
    Cost after iteration 1478: 0.186230
    Cost after iteration 1479: 0.186209
    Cost after iteration 1480: 0.186189
    Cost after iteration 1481: 0.186169
    Cost after iteration 1482: 0.186149
    Cost after iteration 1483: 0.186132
    Cost after iteration 1484: 0.186116
    Cost after iteration 1485: 0.186104
    Cost after iteration 1486: 0.186098
    Cost after iteration 1487: 0.186104
    Cost after iteration 1488: 0.186129
    Cost after iteration 1489: 0.186197
    Cost after iteration 1490: 0.186329
    Cost after iteration 1491: 0.186625
    Cost after iteration 1492: 0.187135
    Cost after iteration 1493: 0.188337
    Cost after iteration 1494: 0.190250
    Cost after iteration 1495: 0.195451
    Cost after iteration 1496: 0.203135
    Cost after iteration 1497: 0.227569
    Cost after iteration 1498: 0.248772
    Cost after iteration 1499: 0.349934
    Cost after iteration 1500: 0.316995
    Cost after iteration 1501: 0.469207
    Cost after iteration 1502: 0.318498
    Cost after iteration 1503: 0.443205
    Cost after iteration 1504: 0.416470
    Cost after iteration 1505: 0.485583
    Cost after iteration 1506: 0.280977
    Cost after iteration 1507: 0.262223
    Cost after iteration 1508: 0.219509
    Cost after iteration 1509: 0.204462
    Cost after iteration 1510: 0.197382
    Cost after iteration 1511: 0.194692
    Cost after iteration 1512: 0.193565
    Cost after iteration 1513: 0.193008
    Cost after iteration 1514: 0.192633
    Cost after iteration 1515: 0.192334
    Cost after iteration 1516: 0.192070
    Cost after iteration 1517: 0.191827
    Cost after iteration 1518: 0.191599
    Cost after iteration 1519: 0.191383
    Cost after iteration 1520: 0.191178
    Cost after iteration 1521: 0.190983
    Cost after iteration 1522: 0.190797
    Cost after iteration 1523: 0.190620
    Cost after iteration 1524: 0.190451
    Cost after iteration 1525: 0.190290
    Cost after iteration 1526: 0.190136
    Cost after iteration 1527: 0.189988
    Cost after iteration 1528: 0.189847
    Cost after iteration 1529: 0.189713
    Cost after iteration 1530: 0.189584
    Cost after iteration 1531: 0.189460
    Cost after iteration 1532: 0.189342
    Cost after iteration 1533: 0.189228
    Cost after iteration 1534: 0.189119
    Cost after iteration 1535: 0.189015
    Cost after iteration 1536: 0.188915
    Cost after iteration 1537: 0.188818
    Cost after iteration 1538: 0.188726
    Cost after iteration 1539: 0.188637
    Cost after iteration 1540: 0.188551
    Cost after iteration 1541: 0.188468
    Cost after iteration 1542: 0.188389
    Cost after iteration 1543: 0.188312
    Cost after iteration 1544: 0.188239
    Cost after iteration 1545: 0.188167
    Cost after iteration 1546: 0.188099
    Cost after iteration 1547: 0.188032
    Cost after iteration 1548: 0.187968
    Cost after iteration 1549: 0.187906
    Cost after iteration 1550: 0.187846
    Cost after iteration 1551: 0.187788
    Cost after iteration 1552: 0.187732
    Cost after iteration 1553: 0.187678
    Cost after iteration 1554: 0.187625
    Cost after iteration 1555: 0.187574
    Cost after iteration 1556: 0.187524
    Cost after iteration 1557: 0.187476
    Cost after iteration 1558: 0.187429
    Cost after iteration 1559: 0.187384
    Cost after iteration 1560: 0.187339
    Cost after iteration 1561: 0.187296
    Cost after iteration 1562: 0.187255
    Cost after iteration 1563: 0.187214
    Cost after iteration 1564: 0.187174
    Cost after iteration 1565: 0.187135
    Cost after iteration 1566: 0.187098
    Cost after iteration 1567: 0.187061
    Cost after iteration 1568: 0.187025
    Cost after iteration 1569: 0.186990
    Cost after iteration 1570: 0.186956
    Cost after iteration 1571: 0.186922
    Cost after iteration 1572: 0.186890
    Cost after iteration 1573: 0.186858
    Cost after iteration 1574: 0.186827
    Cost after iteration 1575: 0.186796
    Cost after iteration 1576: 0.186766
    Cost after iteration 1577: 0.186737
    Cost after iteration 1578: 0.186708
    Cost after iteration 1579: 0.186680
    Cost after iteration 1580: 0.186652
    Cost after iteration 1581: 0.186625
    Cost after iteration 1582: 0.186599
    Cost after iteration 1583: 0.186573
    Cost after iteration 1584: 0.186547
    Cost after iteration 1585: 0.186522
    Cost after iteration 1586: 0.186497
    Cost after iteration 1587: 0.186473
    Cost after iteration 1588: 0.186449
    Cost after iteration 1589: 0.186426
    Cost after iteration 1590: 0.186403
    Cost after iteration 1591: 0.186380
    Cost after iteration 1592: 0.186358
    Cost after iteration 1593: 0.186337
    Cost after iteration 1594: 0.186316
    Cost after iteration 1595: 0.186296
    Cost after iteration 1596: 0.186278
    Cost after iteration 1597: 0.186262
    Cost after iteration 1598: 0.186249
    Cost after iteration 1599: 0.186244
    Cost after iteration 1600: 0.186249
    Cost after iteration 1601: 0.186277
    Cost after iteration 1602: 0.186338
    Cost after iteration 1603: 0.186478
    Cost after iteration 1604: 0.186723
    Cost after iteration 1605: 0.187270
    Cost after iteration 1606: 0.188141
    Cost after iteration 1607: 0.190299
    Cost after iteration 1608: 0.193484
    Cost after iteration 1609: 0.202795
    Cost after iteration 1610: 0.214892
    Cost after iteration 1611: 0.256251
    Cost after iteration 1612: 0.269337
    Cost after iteration 1613: 0.393333
    Cost after iteration 1614: 0.300822
    Cost after iteration 1615: 0.391906
    Cost after iteration 1616: 0.288058
    Cost after iteration 1617: 0.374919
    Cost after iteration 1618: 0.376528
    Cost after iteration 1619: 0.462589
    Cost after iteration 1620: 0.299575
    Cost after iteration 1621: 0.294129
    Cost after iteration 1622: 0.229662
    Cost after iteration 1623: 0.210291
    Cost after iteration 1624: 0.199425
    Cost after iteration 1625: 0.195239
    Cost after iteration 1626: 0.193563
    Cost after iteration 1627: 0.192848
    Cost after iteration 1628: 0.192424
    Cost after iteration 1629: 0.192118
    Cost after iteration 1630: 0.191859
    Cost after iteration 1631: 0.191626
    Cost after iteration 1632: 0.191410
    Cost after iteration 1633: 0.191207
    Cost after iteration 1634: 0.191015
    Cost after iteration 1635: 0.190831
    Cost after iteration 1636: 0.190657
    Cost after iteration 1637: 0.190490
    Cost after iteration 1638: 0.190331
    Cost after iteration 1639: 0.190180
    Cost after iteration 1640: 0.190034
    Cost after iteration 1641: 0.189895
    Cost after iteration 1642: 0.189762
    Cost after iteration 1643: 0.189635
    Cost after iteration 1644: 0.189513
    Cost after iteration 1645: 0.189396
    Cost after iteration 1646: 0.189284
    Cost after iteration 1647: 0.189176
    Cost after iteration 1648: 0.189073
    Cost after iteration 1649: 0.188974
    Cost after iteration 1650: 0.188878
    Cost after iteration 1651: 0.188786
    Cost after iteration 1652: 0.188698
    Cost after iteration 1653: 0.188613
    Cost after iteration 1654: 0.188532
    Cost after iteration 1655: 0.188453
    Cost after iteration 1656: 0.188377
    Cost after iteration 1657: 0.188304
    Cost after iteration 1658: 0.188233
    Cost after iteration 1659: 0.188165
    Cost after iteration 1660: 0.188099
    Cost after iteration 1661: 0.188035
    Cost after iteration 1662: 0.187973
    Cost after iteration 1663: 0.187914
    Cost after iteration 1664: 0.187856
    Cost after iteration 1665: 0.187800
    Cost after iteration 1666: 0.187746
    Cost after iteration 1667: 0.187693
    Cost after iteration 1668: 0.187643
    Cost after iteration 1669: 0.187593
    Cost after iteration 1670: 0.187545
    Cost after iteration 1671: 0.187499
    Cost after iteration 1672: 0.187453
    Cost after iteration 1673: 0.187409
    Cost after iteration 1674: 0.187366
    Cost after iteration 1675: 0.187325
    Cost after iteration 1676: 0.187284
    Cost after iteration 1677: 0.187245
    Cost after iteration 1678: 0.187206
    Cost after iteration 1679: 0.187169
    Cost after iteration 1680: 0.187132
    Cost after iteration 1681: 0.187096
    Cost after iteration 1682: 0.187061
    Cost after iteration 1683: 0.187027
    Cost after iteration 1684: 0.186994
    Cost after iteration 1685: 0.186961
    Cost after iteration 1686: 0.186930
    Cost after iteration 1687: 0.186898
    Cost after iteration 1688: 0.186868
    Cost after iteration 1689: 0.186838
    Cost after iteration 1690: 0.186809
    Cost after iteration 1691: 0.186780
    Cost after iteration 1692: 0.186752
    Cost after iteration 1693: 0.186725
    Cost after iteration 1694: 0.186698
    Cost after iteration 1695: 0.186671
    Cost after iteration 1696: 0.186646
    Cost after iteration 1697: 0.186620
    Cost after iteration 1698: 0.186595
    Cost after iteration 1699: 0.186571
    Cost after iteration 1700: 0.186547
    Cost after iteration 1701: 0.186523
    Cost after iteration 1702: 0.186501
    Cost after iteration 1703: 0.186479
    Cost after iteration 1704: 0.186458
    Cost after iteration 1705: 0.186439
    Cost after iteration 1706: 0.186422
    Cost after iteration 1707: 0.186408
    Cost after iteration 1708: 0.186401
    Cost after iteration 1709: 0.186405
    Cost after iteration 1710: 0.186425
    Cost after iteration 1711: 0.186481
    Cost after iteration 1712: 0.186585
    Cost after iteration 1713: 0.186815
    Cost after iteration 1714: 0.187186
    Cost after iteration 1715: 0.188034
    Cost after iteration 1716: 0.189293
    Cost after iteration 1717: 0.192565
    Cost after iteration 1718: 0.197086
    Cost after iteration 1719: 0.210947
    Cost after iteration 1720: 0.225748
    Cost after iteration 1721: 0.282516
    Cost after iteration 1722: 0.280563
    Cost after iteration 1723: 0.404165
    Cost after iteration 1724: 0.279590
    Cost after iteration 1725: 0.326927
    Cost after iteration 1726: 0.259695
    Cost after iteration 1727: 0.308516
    Cost after iteration 1728: 0.321737
    Cost after iteration 1729: 0.418994
    Cost after iteration 1730: 0.329837
    Cost after iteration 1731: 0.347643
    Cost after iteration 1732: 0.251388
    Cost after iteration 1733: 0.228797
    Cost after iteration 1734: 0.206551
    Cost after iteration 1735: 0.197873
    Cost after iteration 1736: 0.194299
    Cost after iteration 1737: 0.193004
    Cost after iteration 1738: 0.192381
    Cost after iteration 1739: 0.192015
    Cost after iteration 1740: 0.191737
    Cost after iteration 1741: 0.191503
    Cost after iteration 1742: 0.191292
    Cost after iteration 1743: 0.191097
    Cost after iteration 1744: 0.190913
    Cost after iteration 1745: 0.190739
    Cost after iteration 1746: 0.190573
    Cost after iteration 1747: 0.190415
    Cost after iteration 1748: 0.190263
    Cost after iteration 1749: 0.190119
    Cost after iteration 1750: 0.189981
    Cost after iteration 1751: 0.189849
    Cost after iteration 1752: 0.189722
    Cost after iteration 1753: 0.189600
    Cost after iteration 1754: 0.189484
    Cost after iteration 1755: 0.189372
    Cost after iteration 1756: 0.189265
    Cost after iteration 1757: 0.189161
    Cost after iteration 1758: 0.189062
    Cost after iteration 1759: 0.188967
    Cost after iteration 1760: 0.188875
    Cost after iteration 1761: 0.188787
    Cost after iteration 1762: 0.188702
    Cost after iteration 1763: 0.188620
    Cost after iteration 1764: 0.188542
    Cost after iteration 1765: 0.188465
    Cost after iteration 1766: 0.188392
    Cost after iteration 1767: 0.188321
    Cost after iteration 1768: 0.188253
    Cost after iteration 1769: 0.188187
    Cost after iteration 1770: 0.188123
    Cost after iteration 1771: 0.188061
    Cost after iteration 1772: 0.188001
    Cost after iteration 1773: 0.187943
    Cost after iteration 1774: 0.187887
    Cost after iteration 1775: 0.187833
    Cost after iteration 1776: 0.187780
    Cost after iteration 1777: 0.187729
    Cost after iteration 1778: 0.187679
    Cost after iteration 1779: 0.187631
    Cost after iteration 1780: 0.187584
    Cost after iteration 1781: 0.187539
    Cost after iteration 1782: 0.187494
    Cost after iteration 1783: 0.187451
    Cost after iteration 1784: 0.187409
    Cost after iteration 1785: 0.187369
    Cost after iteration 1786: 0.187329
    Cost after iteration 1787: 0.187290
    Cost after iteration 1788: 0.187253
    Cost after iteration 1789: 0.187216
    Cost after iteration 1790: 0.187180
    Cost after iteration 1791: 0.187145
    Cost after iteration 1792: 0.187110
    Cost after iteration 1793: 0.187077
    Cost after iteration 1794: 0.187044
    Cost after iteration 1795: 0.187012
    Cost after iteration 1796: 0.186981
    Cost after iteration 1797: 0.186951
    Cost after iteration 1798: 0.186921
    Cost after iteration 1799: 0.186891
    Cost after iteration 1800: 0.186863
    Cost after iteration 1801: 0.186835
    Cost after iteration 1802: 0.186807
    Cost after iteration 1803: 0.186781
    Cost after iteration 1804: 0.186754
    Cost after iteration 1805: 0.186729
    Cost after iteration 1806: 0.186704
    Cost after iteration 1807: 0.186680
    Cost after iteration 1808: 0.186658
    Cost after iteration 1809: 0.186637
    Cost after iteration 1810: 0.186618
    Cost after iteration 1811: 0.186603
    Cost after iteration 1812: 0.186594
    Cost after iteration 1813: 0.186595
    Cost after iteration 1814: 0.186610
    Cost after iteration 1815: 0.186656
    Cost after iteration 1816: 0.186740
    Cost after iteration 1817: 0.186926
    Cost after iteration 1818: 0.187215
    Cost after iteration 1819: 0.187859
    Cost after iteration 1820: 0.188781
    Cost after iteration 1821: 0.191088
    Cost after iteration 1822: 0.194148
    Cost after iteration 1823: 0.203181
    Cost after iteration 1824: 0.213549
    Cost after iteration 1825: 0.249404
    Cost after iteration 1826: 0.257368
    Cost after iteration 1827: 0.354098
    Cost after iteration 1828: 0.281370
    Cost after iteration 1829: 0.352758
    Cost after iteration 1830: 0.245366
    Cost after iteration 1831: 0.258592
    Cost after iteration 1832: 0.245045
    Cost after iteration 1833: 0.280156
    Cost after iteration 1834: 0.289146
    Cost after iteration 1835: 0.361610
    Cost after iteration 1836: 0.312375
    Cost after iteration 1837: 0.339422
    Cost after iteration 1838: 0.255285
    Cost after iteration 1839: 0.238099
    Cost after iteration 1840: 0.210763
    Cost after iteration 1841: 0.199842
    Cost after iteration 1842: 0.194686
    Cost after iteration 1843: 0.192814
    Cost after iteration 1844: 0.191978
    Cost after iteration 1845: 0.191558
    Cost after iteration 1846: 0.191269
    Cost after iteration 1847: 0.191045
    Cost after iteration 1848: 0.190850
    Cost after iteration 1849: 0.190674
    Cost after iteration 1850: 0.190510
    Cost after iteration 1851: 0.190356
    Cost after iteration 1852: 0.190209
    Cost after iteration 1853: 0.190070
    Cost after iteration 1854: 0.189937
    Cost after iteration 1855: 0.189809
    Cost after iteration 1856: 0.189687
    Cost after iteration 1857: 0.189570
    Cost after iteration 1858: 0.189458
    Cost after iteration 1859: 0.189351
    Cost after iteration 1860: 0.189248
    Cost after iteration 1861: 0.189148
    Cost after iteration 1862: 0.189053
    Cost after iteration 1863: 0.188961
    Cost after iteration 1864: 0.188873
    Cost after iteration 1865: 0.188788
    Cost after iteration 1866: 0.188706
    Cost after iteration 1867: 0.188627
    Cost after iteration 1868: 0.188551
    Cost after iteration 1869: 0.188477
    Cost after iteration 1870: 0.188406
    Cost after iteration 1871: 0.188338
    Cost after iteration 1872: 0.188272
    Cost after iteration 1873: 0.188208
    Cost after iteration 1874: 0.188146
    Cost after iteration 1875: 0.188086
    Cost after iteration 1876: 0.188028
    Cost after iteration 1877: 0.187972
    Cost after iteration 1878: 0.187917
    Cost after iteration 1879: 0.187864
    Cost after iteration 1880: 0.187813
    Cost after iteration 1881: 0.187763
    Cost after iteration 1882: 0.187715
    Cost after iteration 1883: 0.187668
    Cost after iteration 1884: 0.187623
    Cost after iteration 1885: 0.187578
    Cost after iteration 1886: 0.187535
    Cost after iteration 1887: 0.187493
    Cost after iteration 1888: 0.187452
    Cost after iteration 1889: 0.187412
    Cost after iteration 1890: 0.187374
    Cost after iteration 1891: 0.187336
    Cost after iteration 1892: 0.187299
    Cost after iteration 1893: 0.187263
    Cost after iteration 1894: 0.187228
    Cost after iteration 1895: 0.187193
    Cost after iteration 1896: 0.187160
    Cost after iteration 1897: 0.187127
    Cost after iteration 1898: 0.187095
    Cost after iteration 1899: 0.187064
    Cost after iteration 1900: 0.187034
    Cost after iteration 1901: 0.187004
    Cost after iteration 1902: 0.186975
    Cost after iteration 1903: 0.186946
    Cost after iteration 1904: 0.186919
    Cost after iteration 1905: 0.186893
    Cost after iteration 1906: 0.186867
    Cost after iteration 1907: 0.186843
    Cost after iteration 1908: 0.186821
    Cost after iteration 1909: 0.186802
    Cost after iteration 1910: 0.186787
    Cost after iteration 1911: 0.186779
    Cost after iteration 1912: 0.186779
    Cost after iteration 1913: 0.186798
    Cost after iteration 1914: 0.186837
    Cost after iteration 1915: 0.186930
    Cost after iteration 1916: 0.187075
    Cost after iteration 1917: 0.187386
    Cost after iteration 1918: 0.187823
    Cost after iteration 1919: 0.188830
    Cost after iteration 1920: 0.190134
    Cost after iteration 1921: 0.193585
    Cost after iteration 1922: 0.197741
    Cost after iteration 1923: 0.210626
    Cost after iteration 1924: 0.222220
    Cost after iteration 1925: 0.267718
    Cost after iteration 1926: 0.262842
    Cost after iteration 1927: 0.353542
    Cost after iteration 1928: 0.265692
    Cost after iteration 1929: 0.306824
    Cost after iteration 1930: 0.232566
    Cost after iteration 1931: 0.239804
    Cost after iteration 1932: 0.228957
    Cost after iteration 1933: 0.243745
    Cost after iteration 1934: 0.247542
    Cost after iteration 1935: 0.284880
    Cost after iteration 1936: 0.279903
    Cost after iteration 1937: 0.326788
    Cost after iteration 1938: 0.276611
    Cost after iteration 1939: 0.286744
    Cost after iteration 1940: 0.235302
    Cost after iteration 1941: 0.220187
    Cost after iteration 1942: 0.203822
    Cost after iteration 1943: 0.196700
    Cost after iteration 1944: 0.193201
    Cost after iteration 1945: 0.191851
    Cost after iteration 1946: 0.191179
    Cost after iteration 1947: 0.190822
    Cost after iteration 1948: 0.190570
    Cost after iteration 1949: 0.190377
    Cost after iteration 1950: 0.190210
    Cost after iteration 1951: 0.190061
    Cost after iteration 1952: 0.189923
    Cost after iteration 1953: 0.189795
    Cost after iteration 1954: 0.189673
    Cost after iteration 1955: 0.189557
    Cost after iteration 1956: 0.189447
    Cost after iteration 1957: 0.189341
    Cost after iteration 1958: 0.189240
    Cost after iteration 1959: 0.189143
    Cost after iteration 1960: 0.189050
    Cost after iteration 1961: 0.188960
    Cost after iteration 1962: 0.188874
    Cost after iteration 1963: 0.188791
    Cost after iteration 1964: 0.188711
    Cost after iteration 1965: 0.188634
    Cost after iteration 1966: 0.188560
    Cost after iteration 1967: 0.188488
    Cost after iteration 1968: 0.188419
    Cost after iteration 1969: 0.188352
    Cost after iteration 1970: 0.188287
    Cost after iteration 1971: 0.188225
    Cost after iteration 1972: 0.188164
    Cost after iteration 1973: 0.188105
    Cost after iteration 1974: 0.188049
    Cost after iteration 1975: 0.187994
    Cost after iteration 1976: 0.187941
    Cost after iteration 1977: 0.187889
    Cost after iteration 1978: 0.187839
    Cost after iteration 1979: 0.187790
    Cost after iteration 1980: 0.187743
    Cost after iteration 1981: 0.187697
    Cost after iteration 1982: 0.187652
    Cost after iteration 1983: 0.187609
    Cost after iteration 1984: 0.187566
    Cost after iteration 1985: 0.187525
    Cost after iteration 1986: 0.187485
    Cost after iteration 1987: 0.187446
    Cost after iteration 1988: 0.187408
    Cost after iteration 1989: 0.187371
    Cost after iteration 1990: 0.187335
    Cost after iteration 1991: 0.187300
    Cost after iteration 1992: 0.187266
    Cost after iteration 1993: 0.187232
    Cost after iteration 1994: 0.187200
    Cost after iteration 1995: 0.187168
    Cost after iteration 1996: 0.187138
    Cost after iteration 1997: 0.187108
    Cost after iteration 1998: 0.187080
    Cost after iteration 1999: 0.187053
    Cost after iteration 2000: 0.187028
    Cost after iteration 2001: 0.187005
    Cost after iteration 2002: 0.186986
    Cost after iteration 2003: 0.186972
    Cost after iteration 2004: 0.186963
    Cost after iteration 2005: 0.186967
    Cost after iteration 2006: 0.186983
    Cost after iteration 2007: 0.187030
    Cost after iteration 2008: 0.187103
    Cost after iteration 2009: 0.187262
    Cost after iteration 2010: 0.187478
    Cost after iteration 2011: 0.187949
    Cost after iteration 2012: 0.188536
    Cost after iteration 2013: 0.189947
    Cost after iteration 2014: 0.191575
    Cost after iteration 2015: 0.196125
    Cost after iteration 2016: 0.200970
    Cost after iteration 2017: 0.216637
    Cost after iteration 2018: 0.226759
    Cost after iteration 2019: 0.274386
    Cost after iteration 2020: 0.259441
    Cost after iteration 2021: 0.333853
    Cost after iteration 2022: 0.253870
    Cost after iteration 2023: 0.281283
    Cost after iteration 2024: 0.225351
    Cost after iteration 2025: 0.230003
    Cost after iteration 2026: 0.219688
    Cost after iteration 2027: 0.226083
    Cost after iteration 2028: 0.225122
    Cost after iteration 2029: 0.238654
    Cost after iteration 2030: 0.240065
    Cost after iteration 2031: 0.265516
    Cost after iteration 2032: 0.257837
    Cost after iteration 2033: 0.285596
    Cost after iteration 2034: 0.254901
    Cost after iteration 2035: 0.261355
    Cost after iteration 2036: 0.229478
    Cost after iteration 2037: 0.218843
    Cost after iteration 2038: 0.204400
    Cost after iteration 2039: 0.197527
    Cost after iteration 2040: 0.193440
    Cost after iteration 2041: 0.191723
    Cost after iteration 2042: 0.190814
    Cost after iteration 2043: 0.190363
    Cost after iteration 2044: 0.190063
    Cost after iteration 2045: 0.189860
    Cost after iteration 2046: 0.189693
    Cost after iteration 2047: 0.189556
    Cost after iteration 2048: 0.189431
    Cost after iteration 2049: 0.189318
    Cost after iteration 2050: 0.189213
    Cost after iteration 2051: 0.189114
    Cost after iteration 2052: 0.189021
    Cost after iteration 2053: 0.188932
    Cost after iteration 2054: 0.188846
    Cost after iteration 2055: 0.188765
    Cost after iteration 2056: 0.188686
    Cost after iteration 2057: 0.188611
    Cost after iteration 2058: 0.188539
    Cost after iteration 2059: 0.188469
    Cost after iteration 2060: 0.188401
    Cost after iteration 2061: 0.188336
    Cost after iteration 2062: 0.188273
    Cost after iteration 2063: 0.188212
    Cost after iteration 2064: 0.188154
    Cost after iteration 2065: 0.188097
    Cost after iteration 2066: 0.188042
    Cost after iteration 2067: 0.187988
    Cost after iteration 2068: 0.187936
    Cost after iteration 2069: 0.187886
    Cost after iteration 2070: 0.187837
    Cost after iteration 2071: 0.187790
    Cost after iteration 2072: 0.187744
    Cost after iteration 2073: 0.187700
    Cost after iteration 2074: 0.187656
    Cost after iteration 2075: 0.187614
    Cost after iteration 2076: 0.187573
    Cost after iteration 2077: 0.187534
    Cost after iteration 2078: 0.187495
    Cost after iteration 2079: 0.187458
    Cost after iteration 2080: 0.187422
    Cost after iteration 2081: 0.187387
    Cost after iteration 2082: 0.187353
    Cost after iteration 2083: 0.187321
    Cost after iteration 2084: 0.187290
    Cost after iteration 2085: 0.187261
    Cost after iteration 2086: 0.187234
    Cost after iteration 2087: 0.187211
    Cost after iteration 2088: 0.187190
    Cost after iteration 2089: 0.187176
    Cost after iteration 2090: 0.187166
    Cost after iteration 2091: 0.187170
    Cost after iteration 2092: 0.187182
    Cost after iteration 2093: 0.187223
    Cost after iteration 2094: 0.187280
    Cost after iteration 2095: 0.187407
    Cost after iteration 2096: 0.187563
    Cost after iteration 2097: 0.187902
    Cost after iteration 2098: 0.188284
    Cost after iteration 2099: 0.189184
    Cost after iteration 2100: 0.190124
    Cost after iteration 2101: 0.192641
    Cost after iteration 2102: 0.195099
    Cost after iteration 2103: 0.202719
    Cost after iteration 2104: 0.209025
    Cost after iteration 2105: 0.231996
    Cost after iteration 2106: 0.235376
    Cost after iteration 2107: 0.286298
    Cost after iteration 2108: 0.253145
    Cost after iteration 2109: 0.302288
    Cost after iteration 2110: 0.237276
    Cost after iteration 2111: 0.249809
    Cost after iteration 2112: 0.218615
    Cost after iteration 2113: 0.221979
    Cost after iteration 2114: 0.214558
    Cost after iteration 2115: 0.218293
    Cost after iteration 2116: 0.216064
    Cost after iteration 2117: 0.222301
    Cost after iteration 2118: 0.221748
    Cost after iteration 2119: 0.231808
    Cost after iteration 2120: 0.230115
    Cost after iteration 2121: 0.243437
    Cost after iteration 2122: 0.236328
    Cost after iteration 2123: 0.247373
    Cost after iteration 2124: 0.232749
    Cost after iteration 2125: 0.234043
    Cost after iteration 2126: 0.218490
    Cost after iteration 2127: 0.212079
    Cost after iteration 2128: 0.202435
    Cost after iteration 2129: 0.197410
    Cost after iteration 2130: 0.193627
    Cost after iteration 2131: 0.191805
    Cost after iteration 2132: 0.190648
    Cost after iteration 2133: 0.190055
    Cost after iteration 2134: 0.189648
    Cost after iteration 2135: 0.189400
    Cost after iteration 2136: 0.189202
    Cost after iteration 2137: 0.189059
    Cost after iteration 2138: 0.188931
    Cost after iteration 2139: 0.188827
    Cost after iteration 2140: 0.188729
    Cost after iteration 2141: 0.188643
    Cost after iteration 2142: 0.188561
    Cost after iteration 2143: 0.188486
    Cost after iteration 2144: 0.188414
    Cost after iteration 2145: 0.188347
    Cost after iteration 2146: 0.188282
    Cost after iteration 2147: 0.188220
    Cost after iteration 2148: 0.188161
    Cost after iteration 2149: 0.188104
    Cost after iteration 2150: 0.188049
    Cost after iteration 2151: 0.187996
    Cost after iteration 2152: 0.187945
    Cost after iteration 2153: 0.187896
    Cost after iteration 2154: 0.187848
    Cost after iteration 2155: 0.187803
    Cost after iteration 2156: 0.187758
    Cost after iteration 2157: 0.187716
    Cost after iteration 2158: 0.187674
    Cost after iteration 2159: 0.187635
    Cost after iteration 2160: 0.187597
    Cost after iteration 2161: 0.187562
    Cost after iteration 2162: 0.187527
    Cost after iteration 2163: 0.187496
    Cost after iteration 2164: 0.187466
    Cost after iteration 2165: 0.187440
    Cost after iteration 2166: 0.187416
    Cost after iteration 2167: 0.187399
    Cost after iteration 2168: 0.187384
    Cost after iteration 2169: 0.187380
    Cost after iteration 2170: 0.187380
    Cost after iteration 2171: 0.187402
    Cost after iteration 2172: 0.187427
    Cost after iteration 2173: 0.187498
    Cost after iteration 2174: 0.187571
    Cost after iteration 2175: 0.187743
    Cost after iteration 2176: 0.187911
    Cost after iteration 2177: 0.188305
    Cost after iteration 2178: 0.188669
    Cost after iteration 2179: 0.189590
    Cost after iteration 2180: 0.190388
    Cost after iteration 2181: 0.192659
    Cost after iteration 2182: 0.194512
    Cost after iteration 2183: 0.200522
    Cost after iteration 2184: 0.204802
    Cost after iteration 2185: 0.220886
    Cost after iteration 2186: 0.224719
    Cost after iteration 2187: 0.259426
    Cost after iteration 2188: 0.240788
    Cost after iteration 2189: 0.280330
    Cost after iteration 2190: 0.235795
    Cost after iteration 2191: 0.253505
    Cost after iteration 2192: 0.221010
    Cost after iteration 2193: 0.226393
    Cost after iteration 2194: 0.215346
    Cost after iteration 2195: 0.219067
    Cost after iteration 2196: 0.214754
    Cost after iteration 2197: 0.219411
    Cost after iteration 2198: 0.217727
    Cost after iteration 2199: 0.224804
    Cost after iteration 2200: 0.223671
    Cost after iteration 2201: 0.233830
    Cost after iteration 2202: 0.230558
    Cost after iteration 2203: 0.241939
    Cost after iteration 2204: 0.233157
    Cost after iteration 2205: 0.239921
    Cost after iteration 2206: 0.226174
    Cost after iteration 2207: 0.224125
    Cost after iteration 2208: 0.211701
    Cost after iteration 2209: 0.205632
    Cost after iteration 2210: 0.198670
    Cost after iteration 2211: 0.195102
    Cost after iteration 2212: 0.192477
    Cost after iteration 2213: 0.191196
    Cost after iteration 2214: 0.190311
    Cost after iteration 2215: 0.189844
    Cost after iteration 2216: 0.189483
    Cost after iteration 2217: 0.189263
    Cost after iteration 2218: 0.189070
    Cost after iteration 2219: 0.188934
    Cost after iteration 2220: 0.188806
    Cost after iteration 2221: 0.188706
    Cost after iteration 2222: 0.188608
    Cost after iteration 2223: 0.188526
    Cost after iteration 2224: 0.188445
    Cost after iteration 2225: 0.188374
    Cost after iteration 2226: 0.188304
    Cost after iteration 2227: 0.188241
    Cost after iteration 2228: 0.188179
    Cost after iteration 2229: 0.188122
    Cost after iteration 2230: 0.188065
    Cost after iteration 2231: 0.188013
    Cost after iteration 2232: 0.187962
    Cost after iteration 2233: 0.187914
    Cost after iteration 2234: 0.187867
    Cost after iteration 2235: 0.187824
    Cost after iteration 2236: 0.187781
    Cost after iteration 2237: 0.187742
    Cost after iteration 2238: 0.187703
    Cost after iteration 2239: 0.187670
    Cost after iteration 2240: 0.187636
    Cost after iteration 2241: 0.187608
    Cost after iteration 2242: 0.187580
    Cost after iteration 2243: 0.187562
    Cost after iteration 2244: 0.187542
    Cost after iteration 2245: 0.187537
    Cost after iteration 2246: 0.187529
    Cost after iteration 2247: 0.187546
    Cost after iteration 2248: 0.187559
    Cost after iteration 2249: 0.187615
    Cost after iteration 2250: 0.187663
    Cost after iteration 2251: 0.187793
    Cost after iteration 2252: 0.187906
    Cost after iteration 2253: 0.188187
    Cost after iteration 2254: 0.188421
    Cost after iteration 2255: 0.189028
    Cost after iteration 2256: 0.189509
    Cost after iteration 2257: 0.190871
    Cost after iteration 2258: 0.191894
    Cost after iteration 2259: 0.195144
    Cost after iteration 2260: 0.197436
    Cost after iteration 2261: 0.205642
    Cost after iteration 2262: 0.209995
    Cost after iteration 2263: 0.229678
    Cost after iteration 2264: 0.228247
    Cost after iteration 2265: 0.262102
    Cost after iteration 2266: 0.236764
    Cost after iteration 2267: 0.265732
    Cost after iteration 2268: 0.227799
    Cost after iteration 2269: 0.239124
    Cost after iteration 2270: 0.217447
    Cost after iteration 2271: 0.221888
    Cost after iteration 2272: 0.213278
    Cost after iteration 2273: 0.216455
    Cost after iteration 2274: 0.212469
    Cost after iteration 2275: 0.216081
    Cost after iteration 2276: 0.214114
    Cost after iteration 2277: 0.219016
    Cost after iteration 2278: 0.217527
    Cost after iteration 2279: 0.223845
    Cost after iteration 2280: 0.221285
    Cost after iteration 2281: 0.227893
    Cost after iteration 2282: 0.222735
    Cost after iteration 2283: 0.226952
    Cost after iteration 2284: 0.219015
    Cost after iteration 2285: 0.218506
    Cost after iteration 2286: 0.210090
    Cost after iteration 2287: 0.206289
    Cost after iteration 2288: 0.200226
    Cost after iteration 2289: 0.196950
    Cost after iteration 2290: 0.193878
    Cost after iteration 2291: 0.192251
    Cost after iteration 2292: 0.190957
    Cost after iteration 2293: 0.190257
    Cost after iteration 2294: 0.189696
    Cost after iteration 2295: 0.189371
    Cost after iteration 2296: 0.189087
    Cost after iteration 2297: 0.188909
    Cost after iteration 2298: 0.188740
    Cost after iteration 2299: 0.188623
    Cost after iteration 2300: 0.188506
    Cost after iteration 2301: 0.188420
    Cost after iteration 2302: 0.188331
    Cost after iteration 2303: 0.188261
    Cost after iteration 2304: 0.188188
    Cost after iteration 2305: 0.188130
    Cost after iteration 2306: 0.188068
    Cost after iteration 2307: 0.188017
    Cost after iteration 2308: 0.187963
    Cost after iteration 2309: 0.187919
    Cost after iteration 2310: 0.187871
    Cost after iteration 2311: 0.187834
    Cost after iteration 2312: 0.187792
    Cost after iteration 2313: 0.187761
    Cost after iteration 2314: 0.187726
    Cost after iteration 2315: 0.187703
    Cost after iteration 2316: 0.187674
    Cost after iteration 2317: 0.187662
    Cost after iteration 2318: 0.187643
    Cost after iteration 2319: 0.187646
    Cost after iteration 2320: 0.187639
    Cost after iteration 2321: 0.187667
    Cost after iteration 2322: 0.187680
    Cost after iteration 2323: 0.187750
    Cost after iteration 2324: 0.187795
    Cost after iteration 2325: 0.187939
    Cost after iteration 2326: 0.188040
    Cost after iteration 2327: 0.188326
    Cost after iteration 2328: 0.188527
    Cost after iteration 2329: 0.189098
    Cost after iteration 2330: 0.189490
    Cost after iteration 2331: 0.190677
    Cost after iteration 2332: 0.191462
    Cost after iteration 2333: 0.194074
    Cost after iteration 2334: 0.195720
    Cost after iteration 2335: 0.201810
    Cost after iteration 2336: 0.205015
    Cost after iteration 2337: 0.219066
    Cost after iteration 2338: 0.220650
    Cost after iteration 2339: 0.246845
    Cost after iteration 2340: 0.232113
    Cost after iteration 2341: 0.260410
    Cost after iteration 2342: 0.229741
    Cost after iteration 2343: 0.245344
    Cost after iteration 2344: 0.221357
    Cost after iteration 2345: 0.228415
    Cost after iteration 2346: 0.217685
    Cost after iteration 2347: 0.222978
    Cost after iteration 2348: 0.218234
    Cost after iteration 2349: 0.224661
    Cost after iteration 2350: 0.222737
    Cost after iteration 2351: 0.232583
    Cost after iteration 2352: 0.231286
    Cost after iteration 2353: 0.246088
    Cost after iteration 2354: 0.240976
    Cost after iteration 2355: 0.257283
    Cost after iteration 2356: 0.242356
    Cost after iteration 2357: 0.249563
    Cost after iteration 2358: 0.229296
    Cost after iteration 2359: 0.224009
    Cost after iteration 2360: 0.209731
    Cost after iteration 2361: 0.202626
    Cost after iteration 2362: 0.196457
    Cost after iteration 2363: 0.193605
    Cost after iteration 2364: 0.191699
    Cost after iteration 2365: 0.190831
    Cost after iteration 2366: 0.190192
    Cost after iteration 2367: 0.189855
    Cost after iteration 2368: 0.189561
    Cost after iteration 2369: 0.189377
    Cost after iteration 2370: 0.189201
    Cost after iteration 2371: 0.189073
    Cost after iteration 2372: 0.188949
    Cost after iteration 2373: 0.188848
    Cost after iteration 2374: 0.188749
    Cost after iteration 2375: 0.188664
    Cost after iteration 2376: 0.188581
    Cost after iteration 2377: 0.188507
    Cost after iteration 2378: 0.188433
    Cost after iteration 2379: 0.188366
    Cost after iteration 2380: 0.188301
    Cost after iteration 2381: 0.188240
    Cost after iteration 2382: 0.188180
    Cost after iteration 2383: 0.188124
    Cost after iteration 2384: 0.188069
    Cost after iteration 2385: 0.188018
    Cost after iteration 2386: 0.187967
    Cost after iteration 2387: 0.187920
    Cost after iteration 2388: 0.187873
    Cost after iteration 2389: 0.187830
    Cost after iteration 2390: 0.187788
    Cost after iteration 2391: 0.187749
    Cost after iteration 2392: 0.187711
    Cost after iteration 2393: 0.187677
    Cost after iteration 2394: 0.187644
    Cost after iteration 2395: 0.187617
    Cost after iteration 2396: 0.187590
    Cost after iteration 2397: 0.187573
    Cost after iteration 2398: 0.187555
    Cost after iteration 2399: 0.187553
    Cost after iteration 2400: 0.187549
    Cost after iteration 2401: 0.187573
    Cost after iteration 2402: 0.187593
    Cost after iteration 2403: 0.187663
    Cost after iteration 2404: 0.187726
    Cost after iteration 2405: 0.187889
    Cost after iteration 2406: 0.188031
    Cost after iteration 2407: 0.188385
    Cost after iteration 2408: 0.188681
    Cost after iteration 2409: 0.189462
    Cost after iteration 2410: 0.190081
    Cost after iteration 2411: 0.191885
    Cost after iteration 2412: 0.193238
    Cost after iteration 2413: 0.197686
    Cost after iteration 2414: 0.200732
    Cost after iteration 2415: 0.212101
    Cost after iteration 2416: 0.216506
    Cost after iteration 2417: 0.242159
    Cost after iteration 2418: 0.233466
    Cost after iteration 2419: 0.269275
    Cost after iteration 2420: 0.235317
    Cost after iteration 2421: 0.257867
    Cost after iteration 2422: 0.222535
    Cost after iteration 2423: 0.229663
    Cost after iteration 2424: 0.213883
    Cost after iteration 2425: 0.216814
    Cost after iteration 2426: 0.210038
    Cost after iteration 2427: 0.212076
    Cost after iteration 2428: 0.208706
    Cost after iteration 2429: 0.210911
    Cost after iteration 2430: 0.209051
    Cost after iteration 2431: 0.211766
    Cost after iteration 2432: 0.210253
    Cost after iteration 2433: 0.213251
    Cost after iteration 2434: 0.211232
    Cost after iteration 2435: 0.213805
    Cost after iteration 2436: 0.210782
    Cost after iteration 2437: 0.212004
    Cost after iteration 2438: 0.208055
    Cost after iteration 2439: 0.207456
    Cost after iteration 2440: 0.203326
    Cost after iteration 2441: 0.201532
    Cost after iteration 2442: 0.198164
    Cost after iteration 2443: 0.196344
    Cost after iteration 2444: 0.194140
    Cost after iteration 2445: 0.192880
    Cost after iteration 2446: 0.191619
    Cost after iteration 2447: 0.190875
    Cost after iteration 2448: 0.190179
    Cost after iteration 2449: 0.189755
    Cost after iteration 2450: 0.189359
    Cost after iteration 2451: 0.189109
    Cost after iteration 2452: 0.188867
    Cost after iteration 2453: 0.188711
    Cost after iteration 2454: 0.188551
    Cost after iteration 2455: 0.188446
    Cost after iteration 2456: 0.188332
    Cost after iteration 2457: 0.188257
    Cost after iteration 2458: 0.188171
    Cost after iteration 2459: 0.188116
    Cost after iteration 2460: 0.188049
    Cost after iteration 2461: 0.188008
    Cost after iteration 2462: 0.187955
    Cost after iteration 2463: 0.187927
    Cost after iteration 2464: 0.187885
    Cost after iteration 2465: 0.187871
    Cost after iteration 2466: 0.187840
    Cost after iteration 2467: 0.187842
    Cost after iteration 2468: 0.187824
    Cost after iteration 2469: 0.187848
    Cost after iteration 2470: 0.187846
    Cost after iteration 2471: 0.187904
    Cost after iteration 2472: 0.187925
    Cost after iteration 2473: 0.188037
    Cost after iteration 2474: 0.188094
    Cost after iteration 2475: 0.188297
    Cost after iteration 2476: 0.188411
    Cost after iteration 2477: 0.188779
    Cost after iteration 2478: 0.188993
    Cost after iteration 2479: 0.189675
    Cost after iteration 2480: 0.190068
    Cost after iteration 2481: 0.191387
    Cost after iteration 2482: 0.192128
    Cost after iteration 2483: 0.194821
    Cost after iteration 2484: 0.196271
    Cost after iteration 2485: 0.202055
    Cost after iteration 2486: 0.204634
    Cost after iteration 2487: 0.216922
    Cost after iteration 2488: 0.218209
    Cost after iteration 2489: 0.239983
    Cost after iteration 2490: 0.229318
    Cost after iteration 2491: 0.253946
    Cost after iteration 2492: 0.230478
    Cost after iteration 2493: 0.247163
    Cost after iteration 2494: 0.227176
    Cost after iteration 2495: 0.238059
    Cost after iteration 2496: 0.228613
    Cost after iteration 2497: 0.240878
    Cost after iteration 2498: 0.238011
    Cost after iteration 2499: 0.259052
    Cost after iteration 2500: 0.257172
    Cost after iteration 2501: 0.291027
    Cost after iteration 2502: 0.271108
    Cost after iteration 2503: 0.295840
    Cost after iteration 2504: 0.253608
    Cost after iteration 2505: 0.250264
    Cost after iteration 2506: 0.221943
    Cost after iteration 2507: 0.210283
    Cost after iteration 2508: 0.199938
    Cost after iteration 2509: 0.195631
    Cost after iteration 2510: 0.192892
    Cost after iteration 2511: 0.191797
    Cost after iteration 2512: 0.190974
    Cost after iteration 2513: 0.190564
    Cost after iteration 2514: 0.190209
    Cost after iteration 2515: 0.189985
    Cost after iteration 2516: 0.189782
    Cost after iteration 2517: 0.189629
    Cost after iteration 2518: 0.189488
    Cost after iteration 2519: 0.189369
    Cost after iteration 2520: 0.189257
    Cost after iteration 2521: 0.189155
    Cost after iteration 2522: 0.189059
    Cost after iteration 2523: 0.188970
    Cost after iteration 2524: 0.188884
    Cost after iteration 2525: 0.188803
    Cost after iteration 2526: 0.188725
    Cost after iteration 2527: 0.188650
    Cost after iteration 2528: 0.188578
    Cost after iteration 2529: 0.188509
    Cost after iteration 2530: 0.188442
    Cost after iteration 2531: 0.188378
    Cost after iteration 2532: 0.188316
    Cost after iteration 2533: 0.188256
    Cost after iteration 2534: 0.188198
    Cost after iteration 2535: 0.188142
    Cost after iteration 2536: 0.188087
    Cost after iteration 2537: 0.188035
    Cost after iteration 2538: 0.187984
    Cost after iteration 2539: 0.187934
    Cost after iteration 2540: 0.187886
    Cost after iteration 2541: 0.187840
    Cost after iteration 2542: 0.187794
    Cost after iteration 2543: 0.187751
    Cost after iteration 2544: 0.187708
    Cost after iteration 2545: 0.187667
    Cost after iteration 2546: 0.187628
    Cost after iteration 2547: 0.187590
    Cost after iteration 2548: 0.187553
    Cost after iteration 2549: 0.187518
    Cost after iteration 2550: 0.187485
    Cost after iteration 2551: 0.187454
    Cost after iteration 2552: 0.187424
    Cost after iteration 2553: 0.187398
    Cost after iteration 2554: 0.187375
    Cost after iteration 2555: 0.187358
    Cost after iteration 2556: 0.187343
    Cost after iteration 2557: 0.187341
    Cost after iteration 2558: 0.187344
    Cost after iteration 2559: 0.187370
    Cost after iteration 2560: 0.187404
    Cost after iteration 2561: 0.187490
    Cost after iteration 2562: 0.187588
    Cost after iteration 2563: 0.187807
    Cost after iteration 2564: 0.188038
    Cost after iteration 2565: 0.188573
    Cost after iteration 2566: 0.189097
    Cost after iteration 2567: 0.190440
    Cost after iteration 2568: 0.191667
    Cost after iteration 2569: 0.195264
    Cost after iteration 2570: 0.198319
    Cost after iteration 2571: 0.208563
    Cost after iteration 2572: 0.214802
    Cost after iteration 2573: 0.241948
    Cost after iteration 2574: 0.237520
    Cost after iteration 2575: 0.283836
    Cost after iteration 2576: 0.244551
    Cost after iteration 2577: 0.277051
    Cost after iteration 2578: 0.227396
    Cost after iteration 2579: 0.235181
    Cost after iteration 2580: 0.214824
    Cost after iteration 2581: 0.217302
    Cost after iteration 2582: 0.210466
    Cost after iteration 2583: 0.212436
    Cost after iteration 2584: 0.209492
    Cost after iteration 2585: 0.212090
    Cost after iteration 2586: 0.210598
    Cost after iteration 2587: 0.214073
    Cost after iteration 2588: 0.212623
    Cost after iteration 2589: 0.216526
    Cost after iteration 2590: 0.214109
    Cost after iteration 2591: 0.217362
    Cost after iteration 2592: 0.213435
    Cost after iteration 2593: 0.214685
    Cost after iteration 2594: 0.209587
    Cost after iteration 2595: 0.208353
    Cost after iteration 2596: 0.203353
    Cost after iteration 2597: 0.200892
    Cost after iteration 2598: 0.197238
    Cost after iteration 2599: 0.195185
    Cost after iteration 2600: 0.193091
    Cost after iteration 2601: 0.191879
    Cost after iteration 2602: 0.190816
    Cost after iteration 2603: 0.190178
    Cost after iteration 2604: 0.189638
    Cost after iteration 2605: 0.189297
    Cost after iteration 2606: 0.189003
    Cost after iteration 2607: 0.188806
    Cost after iteration 2608: 0.188627
    Cost after iteration 2609: 0.188500
    Cost after iteration 2610: 0.188380
    Cost after iteration 2611: 0.188290
    Cost after iteration 2612: 0.188201
    Cost after iteration 2613: 0.188132
    Cost after iteration 2614: 0.188061
    Cost after iteration 2615: 0.188005
    Cost after iteration 2616: 0.187947
    Cost after iteration 2617: 0.187900
    Cost after iteration 2618: 0.187850
    Cost after iteration 2619: 0.187810
    Cost after iteration 2620: 0.187768
    Cost after iteration 2621: 0.187735
    Cost after iteration 2622: 0.187699
    Cost after iteration 2623: 0.187673
    Cost after iteration 2624: 0.187643
    Cost after iteration 2625: 0.187627
    Cost after iteration 2626: 0.187605
    Cost after iteration 2627: 0.187600
    Cost after iteration 2628: 0.187588
    Cost after iteration 2629: 0.187602
    Cost after iteration 2630: 0.187605
    Cost after iteration 2631: 0.187647
    Cost after iteration 2632: 0.187672
    Cost after iteration 2633: 0.187763
    Cost after iteration 2634: 0.187826
    Cost after iteration 2635: 0.188004
    Cost after iteration 2636: 0.188131
    Cost after iteration 2637: 0.188476
    Cost after iteration 2638: 0.188719
    Cost after iteration 2639: 0.189400
    Cost after iteration 2640: 0.189864
    Cost after iteration 2641: 0.191271
    Cost after iteration 2642: 0.192189
    Cost after iteration 2643: 0.195272
    Cost after iteration 2644: 0.197172
    Cost after iteration 2645: 0.204289
    Cost after iteration 2646: 0.207715
    Cost after iteration 2647: 0.223609
    Cost after iteration 2648: 0.223758
    Cost after iteration 2649: 0.251224
    Cost after iteration 2650: 0.233916
    Cost after iteration 2651: 0.260931
    Cost after iteration 2652: 0.231557
    Cost after iteration 2653: 0.246863
    Cost after iteration 2654: 0.227166
    Cost after iteration 2655: 0.237614
    Cost after iteration 2656: 0.230330
    Cost after iteration 2657: 0.244591
    Cost after iteration 2658: 0.243571
    Cost after iteration 2659: 0.270433
    Cost after iteration 2660: 0.266105
    Cost after iteration 2661: 0.303481
    Cost after iteration 2662: 0.271658
    Cost after iteration 2663: 0.288157
    Cost after iteration 2664: 0.243671
    Cost after iteration 2665: 0.233567
    Cost after iteration 2666: 0.212101
    Cost after iteration 2667: 0.202593
    Cost after iteration 2668: 0.196050
    Cost after iteration 2669: 0.193481
    Cost after iteration 2670: 0.191866
    Cost after iteration 2671: 0.191167
    Cost after iteration 2672: 0.190622
    Cost after iteration 2673: 0.190317
    Cost after iteration 2674: 0.190052
    Cost after iteration 2675: 0.189866
    Cost after iteration 2676: 0.189698
    Cost after iteration 2677: 0.189562
    Cost after iteration 2678: 0.189436
    Cost after iteration 2679: 0.189324
    Cost after iteration 2680: 0.189219
    Cost after iteration 2681: 0.189122
    Cost after iteration 2682: 0.189030
    Cost after iteration 2683: 0.188943
    Cost after iteration 2684: 0.188859
    Cost after iteration 2685: 0.188780
    Cost after iteration 2686: 0.188703
    Cost after iteration 2687: 0.188630
    Cost after iteration 2688: 0.188559
    Cost after iteration 2689: 0.188491
    Cost after iteration 2690: 0.188425
    Cost after iteration 2691: 0.188362
    Cost after iteration 2692: 0.188300
    Cost after iteration 2693: 0.188241
    Cost after iteration 2694: 0.188184
    Cost after iteration 2695: 0.188128
    Cost after iteration 2696: 0.188074
    Cost after iteration 2697: 0.188022
    Cost after iteration 2698: 0.187972
    Cost after iteration 2699: 0.187923
    Cost after iteration 2700: 0.187875
    Cost after iteration 2701: 0.187829
    Cost after iteration 2702: 0.187785
    Cost after iteration 2703: 0.187741
    Cost after iteration 2704: 0.187699
    Cost after iteration 2705: 0.187659
    Cost after iteration 2706: 0.187619
    Cost after iteration 2707: 0.187582
    Cost after iteration 2708: 0.187545
    Cost after iteration 2709: 0.187510
    Cost after iteration 2710: 0.187477
    Cost after iteration 2711: 0.187446
    Cost after iteration 2712: 0.187417
    Cost after iteration 2713: 0.187391
    Cost after iteration 2714: 0.187367
    Cost after iteration 2715: 0.187349
    Cost after iteration 2716: 0.187335
    Cost after iteration 2717: 0.187332
    Cost after iteration 2718: 0.187334
    Cost after iteration 2719: 0.187358
    Cost after iteration 2720: 0.187392
    Cost after iteration 2721: 0.187476
    Cost after iteration 2722: 0.187574
    Cost after iteration 2723: 0.187791
    Cost after iteration 2724: 0.188024
    Cost after iteration 2725: 0.188561
    Cost after iteration 2726: 0.189096
    Cost after iteration 2727: 0.190460
    Cost after iteration 2728: 0.191726
    Cost after iteration 2729: 0.195428
    Cost after iteration 2730: 0.198617
    Cost after iteration 2731: 0.209292
    Cost after iteration 2732: 0.215763
    Cost after iteration 2733: 0.244226
    Cost after iteration 2734: 0.238877
    Cost after iteration 2735: 0.286710
    Cost after iteration 2736: 0.245042
    Cost after iteration 2737: 0.277071
    Cost after iteration 2738: 0.226706
    Cost after iteration 2739: 0.233801
    Cost after iteration 2740: 0.214177
    Cost after iteration 2741: 0.216343
    Cost after iteration 2742: 0.209725
    Cost after iteration 2743: 0.211404
    Cost after iteration 2744: 0.208521
    Cost after iteration 2745: 0.210734
    Cost after iteration 2746: 0.209223
    Cost after iteration 2747: 0.212108
    Cost after iteration 2748: 0.210646
    Cost after iteration 2749: 0.213721
    Cost after iteration 2750: 0.211468
    Cost after iteration 2751: 0.213829
    Cost after iteration 2752: 0.210427
    Cost after iteration 2753: 0.211127
    Cost after iteration 2754: 0.206910
    Cost after iteration 2755: 0.205766
    Cost after iteration 2756: 0.201686
    Cost after iteration 2757: 0.199681
    Cost after iteration 2758: 0.196628
    Cost after iteration 2759: 0.194904
    Cost after iteration 2760: 0.193044
    Cost after iteration 2761: 0.191946
    Cost after iteration 2762: 0.190927
    Cost after iteration 2763: 0.190301
    Cost after iteration 2764: 0.189749
    Cost after iteration 2765: 0.189394
    Cost after iteration 2766: 0.189079
    Cost after iteration 2767: 0.188867
    Cost after iteration 2768: 0.188673
    Cost after iteration 2769: 0.188535
    Cost after iteration 2770: 0.188405
    Cost after iteration 2771: 0.188308
    Cost after iteration 2772: 0.188213
    Cost after iteration 2773: 0.188141
    Cost after iteration 2774: 0.188067
    Cost after iteration 2775: 0.188010
    Cost after iteration 2776: 0.187950
    Cost after iteration 2777: 0.187904
    Cost after iteration 2778: 0.187854
    Cost after iteration 2779: 0.187817
    Cost after iteration 2780: 0.187775
    Cost after iteration 2781: 0.187746
    Cost after iteration 2782: 0.187712
    Cost after iteration 2783: 0.187692
    Cost after iteration 2784: 0.187666
    Cost after iteration 2785: 0.187657
    Cost after iteration 2786: 0.187640
    Cost after iteration 2787: 0.187647
    Cost after iteration 2788: 0.187643
    Cost after iteration 2789: 0.187674
    Cost after iteration 2790: 0.187689
    Cost after iteration 2791: 0.187758
    Cost after iteration 2792: 0.187802
    Cost after iteration 2793: 0.187938
    Cost after iteration 2794: 0.188031
    Cost after iteration 2795: 0.188288
    Cost after iteration 2796: 0.188465
    Cost after iteration 2797: 0.188954
    Cost after iteration 2798: 0.189284
    Cost after iteration 2799: 0.190250
    Cost after iteration 2800: 0.190877
    Cost after iteration 2801: 0.192884
    Cost after iteration 2802: 0.194132
    Cost after iteration 2803: 0.198561
    Cost after iteration 2804: 0.201058
    Cost after iteration 2805: 0.211114
    Cost after iteration 2806: 0.214333
    Cost after iteration 2807: 0.234826
    Cost after iteration 2808: 0.229221
    Cost after iteration 2809: 0.257729
    Cost after iteration 2810: 0.234450
    Cost after iteration 2811: 0.256573
    Cost after iteration 2812: 0.230837
    Cost after iteration 2813: 0.244079
    Cost after iteration 2814: 0.231395
    Cost after iteration 2815: 0.245456
    Cost after iteration 2816: 0.242299
    Cost after iteration 2817: 0.267622
    Cost after iteration 2818: 0.265547
    Cost after iteration 2819: 0.306177
    Cost after iteration 2820: 0.278686
    Cost after iteration 2821: 0.303144
    Cost after iteration 2822: 0.252966
    Cost after iteration 2823: 0.245583
    Cost after iteration 2824: 0.218180
    Cost after iteration 2825: 0.206698
    Cost after iteration 2826: 0.198009
    Cost after iteration 2827: 0.194591
    Cost after iteration 2828: 0.192448
    Cost after iteration 2829: 0.191568
    Cost after iteration 2830: 0.190894
    Cost after iteration 2831: 0.190536
    Cost after iteration 2832: 0.190231
    Cost after iteration 2833: 0.190025
    Cost after iteration 2834: 0.189842
    Cost after iteration 2835: 0.189696
    Cost after iteration 2836: 0.189562
    Cost after iteration 2837: 0.189445
    Cost after iteration 2838: 0.189335
    Cost after iteration 2839: 0.189234
    Cost after iteration 2840: 0.189137
    Cost after iteration 2841: 0.189047
    Cost after iteration 2842: 0.188960
    Cost after iteration 2843: 0.188878
    Cost after iteration 2844: 0.188798
    Cost after iteration 2845: 0.188722
    Cost after iteration 2846: 0.188649
    Cost after iteration 2847: 0.188578
    Cost after iteration 2848: 0.188510
    Cost after iteration 2849: 0.188445
    Cost after iteration 2850: 0.188381
    Cost after iteration 2851: 0.188320
    Cost after iteration 2852: 0.188260
    Cost after iteration 2853: 0.188203
    Cost after iteration 2854: 0.188147
    Cost after iteration 2855: 0.188093
    Cost after iteration 2856: 0.188041
    Cost after iteration 2857: 0.187990
    Cost after iteration 2858: 0.187941
    Cost after iteration 2859: 0.187893
    Cost after iteration 2860: 0.187847
    Cost after iteration 2861: 0.187802
    Cost after iteration 2862: 0.187758
    Cost after iteration 2863: 0.187716
    Cost after iteration 2864: 0.187674
    Cost after iteration 2865: 0.187634
    Cost after iteration 2866: 0.187596
    Cost after iteration 2867: 0.187558
    Cost after iteration 2868: 0.187522
    Cost after iteration 2869: 0.187488
    Cost after iteration 2870: 0.187454
    Cost after iteration 2871: 0.187423
    Cost after iteration 2872: 0.187393
    Cost after iteration 2873: 0.187365
    Cost after iteration 2874: 0.187340
    Cost after iteration 2875: 0.187319
    Cost after iteration 2876: 0.187301
    Cost after iteration 2877: 0.187292
    Cost after iteration 2878: 0.187287
    Cost after iteration 2879: 0.187300
    Cost after iteration 2880: 0.187321
    Cost after iteration 2881: 0.187379
    Cost after iteration 2882: 0.187451
    Cost after iteration 2883: 0.187612
    Cost after iteration 2884: 0.187795
    Cost after iteration 2885: 0.188202
    Cost after iteration 2886: 0.188630
    Cost after iteration 2887: 0.189673
    Cost after iteration 2888: 0.190688
    Cost after iteration 2889: 0.193513
    Cost after iteration 2890: 0.196082
    Cost after iteration 2891: 0.204308
    Cost after iteration 2892: 0.210406
    Cost after iteration 2893: 0.233801
    Cost after iteration 2894: 0.234841
    Cost after iteration 2895: 0.282196
    Cost after iteration 2896: 0.248423
    Cost after iteration 2897: 0.290048
    Cost after iteration 2898: 0.232762
    Cost after iteration 2899: 0.243455
    Cost after iteration 2900: 0.216134
    Cost after iteration 2901: 0.218646
    Cost after iteration 2902: 0.210744
    Cost after iteration 2903: 0.212550
    Cost after iteration 2904: 0.209337
    Cost after iteration 2905: 0.211764
    Cost after iteration 2906: 0.210236
    Cost after iteration 2907: 0.213571
    Cost after iteration 2908: 0.212182
    Cost after iteration 2909: 0.215987
    Cost after iteration 2910: 0.213694
    Cost after iteration 2911: 0.216931
    Cost after iteration 2912: 0.213172
    Cost after iteration 2913: 0.214517
    Cost after iteration 2914: 0.209570
    Cost after iteration 2915: 0.208481
    Cost after iteration 2916: 0.203543
    Cost after iteration 2917: 0.201153
    Cost after iteration 2918: 0.197472
    Cost after iteration 2919: 0.195403
    Cost after iteration 2920: 0.193257
    Cost after iteration 2921: 0.192009
    Cost after iteration 2922: 0.190909
    Cost after iteration 2923: 0.190244
    Cost after iteration 2924: 0.189686
    Cost after iteration 2925: 0.189330
    Cost after iteration 2926: 0.189027
    Cost after iteration 2927: 0.188822
    Cost after iteration 2928: 0.188640
    Cost after iteration 2929: 0.188509
    Cost after iteration 2930: 0.188388
    Cost after iteration 2931: 0.188295
    Cost after iteration 2932: 0.188206
    Cost after iteration 2933: 0.188135
    Cost after iteration 2934: 0.188064
    Cost after iteration 2935: 0.188007
    Cost after iteration 2936: 0.187948
    Cost after iteration 2937: 0.187900
    Cost after iteration 2938: 0.187850
    Cost after iteration 2939: 0.187809
    Cost after iteration 2940: 0.187765
    Cost after iteration 2941: 0.187730
    Cost after iteration 2942: 0.187692
    Cost after iteration 2943: 0.187664
    Cost after iteration 2944: 0.187632
    Cost after iteration 2945: 0.187610
    Cost after iteration 2946: 0.187585
    Cost after iteration 2947: 0.187573
    Cost after iteration 2948: 0.187555
    Cost after iteration 2949: 0.187556
    Cost after iteration 2950: 0.187551
    Cost after iteration 2951: 0.187572
    Cost after iteration 2952: 0.187584
    Cost after iteration 2953: 0.187639
    Cost after iteration 2954: 0.187679
    Cost after iteration 2955: 0.187794
    Cost after iteration 2956: 0.187880
    Cost after iteration 2957: 0.188107
    Cost after iteration 2958: 0.188276
    Cost after iteration 2959: 0.188722
    Cost after iteration 2960: 0.189043
    Cost after iteration 2961: 0.189947
    Cost after iteration 2962: 0.190571
    Cost after iteration 2963: 0.192499
    Cost after iteration 2964: 0.193769
    Cost after iteration 2965: 0.198149
    Cost after iteration 2966: 0.200778
    Cost after iteration 2967: 0.211050
    Cost after iteration 2968: 0.214633
    Cost after iteration 2969: 0.236247
    Cost after iteration 2970: 0.230264
    Cost after iteration 2971: 0.260715
    Cost after iteration 2972: 0.235199
    Cost after iteration 2973: 0.258052
    Cost after iteration 2974: 0.229922
    Cost after iteration 2975: 0.242289
    Cost after iteration 2976: 0.228847
    Cost after iteration 2977: 0.240945
    Cost after iteration 2978: 0.237365
    Cost after iteration 2979: 0.258512
    Cost after iteration 2980: 0.257995
    Cost after iteration 2981: 0.295241
    Cost after iteration 2982: 0.276229
    Cost after iteration 2983: 0.306092
    Cost after iteration 2984: 0.258751
    Cost after iteration 2985: 0.256821
    Cost after iteration 2986: 0.224284
    Cost after iteration 2987: 0.211605
    Cost after iteration 2988: 0.200382
    Cost after iteration 2989: 0.195728
    Cost after iteration 2990: 0.192924
    Cost after iteration 2991: 0.191809
    Cost after iteration 2992: 0.191014
    Cost after iteration 2993: 0.190610
    Cost after iteration 2994: 0.190273
    Cost after iteration 2995: 0.190053
    Cost after iteration 2996: 0.189858
    Cost after iteration 2997: 0.189707
    Cost after iteration 2998: 0.189568
    Cost after iteration 2999: 0.189449
    W1 = [[ 2.45242984 -2.06948032]
     [ 2.08048766 -1.88677342]]
    b1 = [[ 6.31177772]
     [-4.85150476]]
    W2 = [[ 7.07784203 -7.20323522]]
    b2 = [[-3.44973187]]


##### __Expected Output__ 
Note: the actual values can be different!

```Python
Cost after iteration 0: 0.693148
Cost after iteration 1: 0.693147
Cost after iteration 2: 0.693147
Cost after iteration 3: 0.693147
Cost after iteration 4: 0.693147
Cost after iteration 5: 0.693147
...
Cost after iteration 2995: 0.209524
Cost after iteration 2996: 0.208025
Cost after iteration 2997: 0.210427
Cost after iteration 2998: 0.208929
Cost after iteration 2999: 0.211306
W1 = [[ 2.14274251 -1.93155541]
 [ 2.20268789 -2.1131799 ]]
b1 = [[-4.83079243]
 [ 6.2845223 ]]
W2 = [[-7.21370685  7.0898022 ]]
b2 = [[-3.48755239]]
```


```python
# Note: 
# Actual values are not checked here in the unit tests (due to random initialization).
w3_unittest.test_nn_model(nn_model)
```

    [92m All tests passed


The final model parameters can be used to find the boundary line and for making predictions. 

<a name='ex08'></a>
### Exercise 8

Computes probabilities using forward propagation, and make classification to 0/1 using 0.5 as the threshold.


```python
# GRADED FUNCTION: predict

def predict(X, parameters):
    """
    Using the learned parameters, predicts a class for each example in X
    
    Arguments:
    parameters -- python dictionary containing your parameters 
    X -- input data of size (n_x, m)
    
    Returns
    predictions -- vector of predictions of our model (blue: 0 / red: 1)
    """
    
    ### START CODE HERE ### (≈ 2 lines of code)
    A2, cache = forward_propagation(X, parameters)
    predictions = A2 > 0.5
    ### END CODE HERE ###
    
    return predictions
```


```python
X_pred = np.array([[2, 8, 2, 8], [2, 8, 8, 2]])
Y_pred = predict(X_pred, parameters)

print(f"Coordinates (in the columns):\n{X_pred}")
print(f"Predictions:\n{Y_pred}")
```

    Coordinates (in the columns):
    [[2 8 2 8]
     [2 8 8 2]]
    Predictions:
    [[ True  True False False]]


##### __Expected Output__ 

```Python
Coordinates (in the columns):
[[2 8 2 8]
 [2 8 8 2]]
Predictions:
[[ True  True False False]]
```


```python
w3_unittest.test_predict(predict)
```

    [92m All tests passed


Let's visualize the boundary line. Do not worry if you don't understand the function `plot_decision_boundary` line by line - it simply makes prediction for some points on the plane and plots them as a contour plot (just two colors - blue and red).


```python
def plot_decision_boundary(predict, parameters, X, Y):
    # Define bounds of the domain.
    min1, max1 = X[0, :].min()-1, X[0, :].max()+1
    min2, max2 = X[1, :].min()-1, X[1, :].max()+1
    # Define the x and y scale.
    x1grid = np.arange(min1, max1, 0.1)
    x2grid = np.arange(min2, max2, 0.1)
    # Create all of the lines and rows of the grid.
    xx, yy = np.meshgrid(x1grid, x2grid)
    # Flatten each grid to a vector.
    r1, r2 = xx.flatten(), yy.flatten()
    r1, r2 = r1.reshape((1, len(r1))), r2.reshape((1, len(r2)))
    # Vertical stack vectors to create x1,x2 input for the model.
    grid = np.vstack((r1,r2))
    # Make predictions for the grid.
    predictions = predict(grid, parameters)
    # Reshape the predictions back into a grid.
    zz = predictions.reshape(xx.shape)
    # Plot the grid of x, y and z values as a surface.
    plt.contourf(xx, yy, zz, cmap=plt.cm.Spectral.reversed())
    plt.scatter(X[0, :], X[1, :], c=Y, cmap=colors.ListedColormap(['blue', 'red']));

# Plot the decision boundary.
plot_decision_boundary(predict, parameters, X, Y)
plt.title("Decision Boundary for hidden layer size " + str(n_h))
```




    Text(0.5, 1.0, 'Decision Boundary for hidden layer size 2')




    
![png](output_80_1.png)
    


That's great, you can see that more complicated classification problems can be solved with two layer neural network!

<a name='4'></a>
## 4 - Optional: Other Dataset

Build a slightly different dataset:


```python
n_samples = 2000
samples, labels = make_blobs(n_samples=n_samples, 
                             centers=([2.5, 3], [6.7, 7.9], [2.1, 7.9], [7.4, 2.8]), 
                             cluster_std=1.1,
                             random_state=0)
labels[(labels == 0)] = 0
labels[(labels == 1)] = 1
labels[(labels == 2) | (labels == 3)] = 1
X_2 = np.transpose(samples)
Y_2 = labels.reshape((1,n_samples))

plt.scatter(X_2[0, :], X_2[1, :], c=Y_2, cmap=colors.ListedColormap(['blue', 'red']));
```


    
![png](output_84_0.png)
    


Notice that when building your neural network, a number of the nodes in the hidden layer could be taken as a parameter. Try to change this parameter and investigate the results:


```python
# parameters_2 = nn_model(X_2, Y_2, n_h=1, num_iterations=3000, learning_rate=1.2, print_cost=False)
parameters_2 = nn_model(X_2, Y_2, n_h=2, num_iterations=3000, learning_rate=1.2, print_cost=False)
# parameters_2 = nn_model(X_2, Y_2, n_h=15, num_iterations=3000, learning_rate=1.2, print_cost=False)

# This function will call predict function 
plot_decision_boundary(predict, parameters_2, X_2, Y_2)
plt.title("Decision Boundary")
```




    Text(0.5, 1.0, 'Decision Boundary')




    
![png](output_86_1.png)
    


You can see that there are some misclassified points - real-world datasets are usually linearly inseparable, and there will be a small percentage of errors. More than that, you do not want to build a model that fits too closely, almost exactly to a particular set of data - it may fail to predict future observations. This problem is known as **overfitting**.

Congrats on finishing this programming assignment!


```python

```
