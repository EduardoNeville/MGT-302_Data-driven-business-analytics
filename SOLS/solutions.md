# Solutions to Homework 1

Eduardo Neville <br>
SCIPER: 314667 <br>

## Question 1:

### Part 1)
#### (a)
The cost function: $J(\theta)=\frac{1}{2 m} \sum_{i=1}^m\left(h_{\Theta}\left(x^{(i)}\right)-y^{(i)}\right)^2$ <br>
My gradient descent of the cost function is 
$$
\theta_j=\theta_j-\alpha \frac{\partial}{\partial \theta_j} J(\theta)
$$
Where,
$$
\begin{aligned}
& \frac{\partial}{\partial \theta} J_\theta=\frac{\partial}{\partial \theta} \frac{1}{2 m} \sum_{i=1}^m\left[h_\theta\left(x_i\right)-y_i\right]^2 \\
& \frac{\partial}{\partial \theta} J_\theta=\frac{1}{m} \sum_{i=1}^m\left(h_\theta\left(x_i\right)-y_i\right) \cdot \frac{\partial}{\partial \theta_j}\left(\theta x_i-y_i\right) \\
& \frac{\partial}{\partial \theta} J_\theta=\frac{1}{m} \sum_{i=1}^m\left[\left(h_\theta\left(x_i\right)-y\right) x_i\right]
\end{aligned}
$$
Therefore,
$$
\theta_j:=\theta_j-\frac{\alpha}{m} \sum_{i=1}^m\left[\left(h_\theta\left(x_i\right)-y_i\right) x_i\right]
$$

### Part 2
#### (a)
Given the previous cost function we shall now calculate the stochastic gradient descent. 
It is a variant of gradient descent that uses random subsamples (mini-batches) of the training data to compute the gradient of the loss function and update the model parameters. Unlike batch gradient descent, which uses the entire training set to compute the gradient, SGD updates the model parameters after processing each mini-batch of the training data. This makes SGD computationally efficient and well-suited for large datasets. SGD is a stochastic optimization algorithm because the gradient computed using a mini-batch is a noisy estimate of the true gradient computed using the entire training set. As a result, the update direction is not exact, but the noise introduces some randomness that can help the optimization process to escape from local minima. 

Given the cost function above. We have that our stochastic gradient is: <br>
$$
\begin{aligned}
&\theta_j:=\theta_j-\eta \nabla Q_i(\theta_j) \\
\end{aligned}
$$
We can express the gradient of the loss function with respect to $\theta_j$ at $i-th$ example as: <br>
$$
\begin{aligned}
&\nabla Q_i\left(\theta_j\right)=\frac{\partial Q_i\left(\theta_j\right)}{\partial \theta_j}
\end{aligned}
$$
Substituting this in the first equation we get:
$$
\begin{aligned}
&\theta_j:=\theta_j-\eta \frac{\partial Q_i\left(\theta_j\right)}{\partial \theta_j}
\end{aligned}
$$
Using the chain rule of differentiation, we can expand the derivative as follows:
$$
\begin{aligned}
&\frac{\partial Q_i\left(\theta_j\right)}{\partial \theta_j}=\frac{\partial Q_i\left(\theta_j\right)}{\partial \widehat{y}^i} \frac{\partial y^i}{\partial z^i} \frac{\partial z^i}{\partial \theta_j}
\end{aligned}
$$
where $z^i = \theta_0 X_0^i + \theta_1 X_1^i + ... + \theta_j X_j^i + ... + \theta_n X_n^i$ and $\widehat{y^i} = \sigma(z^i)$.

Taking the derivative of $z^i$ with respect to $\theta_j$, we get:
$$
\frac{\partial z^i}{\partial \theta_j}=X_j^i
$$
Taking the derivative of $\widehat{y^i}$ with respect to $z^i$ we get:
$$
\frac{\partial \widehat{y^i}}{\partial z^i}=\widehat{y^i}\left(1-\widehat{y^i}\right)
$$
Finally, taking the derivative of $Q_i(\theta_j)$ with respect to $\widehat{y^i}$, we get:
$$
\frac{\partial Q_i\left(\theta_j\right)}{\partial y^i}=-\left(y^i-\widehat{y^i}\right)
$$
Substituting these back into the second equation, we get:
$$
\theta_j:=\theta_j-\eta\left(y^i-\widehat{y^i}\right) \widehat{y^i}\left(1-\widehat{y^i}\right) X_j^i
$$
where $\alpha = \eta\widehat{y^i}(1-\widehat{y^i})$ is the learning rate.
giving us the following equation:
$$
\theta_j=\theta_j-\alpha\left(\widehat{y^i}-y^i\right) X_j^i
$$

## Question 2

To find the first principal component of M, we need to follow these steps:

* Compute the covariance matrix of M
* Compute the eigenvalues and eigenvectors of the covariance matrix
* Sort the eigenvalues in descending order and choose the eigenvector corresponding to the largest eigenvalue as the first principal component.
Let's compute each step:

We first need to find the covariance matrix, to do so we must first calculate the mean of each input data: <br>

$$
\bar{x}=(1+2+0)/3 = 1 \\
\bar{y}=(2+1+0)/3 = 1 \\
\bar{z}=(0+0+0)/3 = 0

$$
Giving us the mean vector: <br>
$$
\mu = \begin{bmatrix}1 & 1 & 0 \end{bmatrix}
$$

We then subtract the means from each column to center the data:

$$
\begin{bmatrix}
1 - 1 & 2 - 1 & 0 - 0 \\
2 - 1 & 1 - 1 & 0 - 0 \\
0 - 1 & 0 - 1 & 0 - 0 \\
\end{bmatrix}
$$

Giving us the following centered data matrix:

$$
X = 
\begin{bmatrix}
0 & 1 & 0   \\
1 & 0 & 0   \\
-1 & -1 & 0 \\
\end{bmatrix}
$$

Next, we compute the covariance matrix as follows 

$$ \Sigma = \frac{1}{n - 1} X^T * X $$
Where $\Sigma$ is the covariance matrix and $X$ is the centered data matrix. This gives us the following matrix:
$$ \Sigma = 
$\begin{bmatrix} 1 & -\frac{1}{2} & 0 \ -\frac{1}{2} & 1 & 0 \ 0 & 0 & 0 \end{bmatrix}$

\begin{bmatrix}
\frac{2}{3} & -\frac{2}{3} & 0 \\
-\frac{2}{3} & \frac{2}{3} & 0 \\
0 & 0 & 0
\end{bmatrix}

