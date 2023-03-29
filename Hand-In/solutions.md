# Solutions to Homework 1

Eduardo Neville <br>
SCIPER: 314667 <br>

And

Åke Janson <br>
SCIPER: 314487 <br>

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
#### (b)
Once implemented we find that the optimal value for $\alpha$ is around 0.01. A plot of the number of iterations and the cost as a function of $\alpha$  can be found in the images named 'GD1_AlphaVsCost.jpg' and 'GD1_AlphaVsIterrations.jpg'.

#### (c)
The plot of the cost as a function of thee number of iterations can be found in the image 'GD2_CostVsIterrations.jpg'. And the algorithm to find it is 'GD2.m'.
#### (d)
Once trained the $\Theta$ is found to be $\begin{bmatrix}
22.29 \\
4.68 \\
2.31 \\
-1.42 \\
5.99 \\
-8.24 \\
\end{bmatrix}$ and the final cost is calculated to be 0.3124, calculated with the test data. The MATLAB for this is in file 'GD3.m'.

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
The implementation of this can be found in the file 'SGD1.py'. When comparing the plots we can see that SGD converges much faster than GD.
### (b)
The step-size of SGD is now chosen as a decreasing sequence $\alpha_t =\frac{b}{1+t}$,  the implementation of this can be found in the file 'SGD2.py'. The parameter t was devided to make the sequence slower. As we can see in files 'SDG1_CostVsIterrations.png' and 'SDG2_CostVsIterrations.png' this algorithm converges faster.

## Question 2

To find the first principal component of M, we need to follow these steps:
* Standardize the data 
* Compute the covariance matrix of M
* Compute the eigenvalues and eigenvectors of the covariance matrix
* Sort the eigenvalues in descending order and choose the eigenvector corresponding to the largest eigenvalue as the first principal component.
Let's compute each step:

We need to start by standardizing the data. To do this we substract the mean of each input data and devide by the standard deviation. Let's start by calculating the mean vector $\mu$.
$$
\bar{a}=\frac{\sum_{i}^{N}a_i}{N}\\
\bar{x}=(1+2+0)/3 = 1 \\
\bar{y}=(2+1+0)/3 = 1 \\
\bar{z}=(0+0+0)/3 = 0
$$
Giving us the mean vector: <br>
$$
\mu = \begin{bmatrix}\bar{x} & \bar{y} & \bar{z} \end{bmatrix} = \begin{bmatrix}1 & 1 & 0 \end{bmatrix}
$$
Next we can calculate the standard deviation of each input data:
$$
\sigma_a = \sqrt{\frac{\sum_{i}^{N}(a_i-\bar{a})^2 }{N-1}}\\
\sigma_x = \sqrt{\frac{(1-1)^2 + (2-1)^2 + (0-1)^2 }{3-1}} = \sqrt{\frac{2}{2}} = 1\\
\sigma_y = \sqrt{\frac{(2-1)^2 + (1-1)^2 + (0-1)^2 }{3-1}} = \sqrt{\frac{2}{2}} = 1\\
\sigma_z = \sqrt{\frac{(0-0)^2 + (0-0)^2 + (0-0)^2 }{3-1}} = \sqrt{\frac{0}{3}} = 0
$$

We then subtract the means and substract by the standard deviation from each column to standardize the data:

$$
a_{standard} = \frac{a-\bar{a}}{\sigma_a}\\
\begin{bmatrix}
\frac{1 - 1}{1} & \frac{2 - 1}{1} & 0 \\
\frac{2 - 1}{1} & \frac{1 - 1}{1} & 0 \\
\frac{0 - 1}{1} & \frac{0 - 1}{1} & 0 \\
\end{bmatrix}
$$

Giving us the following standardized data matrix:

$$
X = 
\begin{bmatrix}
0 & 1 & 0   \\
1 & 0 & 0   \\
-1 & -1 & 0 \\
\end{bmatrix}
$$

Next, we compute the covariance matrix as follows 
$$
cov(a,b) = \frac{1}{N-1}\sum_{i=1}^{N}(a_i-\bar{a})(b_i-\bar{b})\\
\Sigma = \begin{bmatrix}
cov(x,x) & cov(x,y) & cov(x,z) \\
cov(y,x) & cov(y,y) & cov(y,z) \\
cov(z,x) & cov(z,y) & cov(z,z)
\end{bmatrix} =  \frac{1}{N - 1} X^T * X 
$$

Where $\Sigma$ is the covariance matrix. This gives us the following matrix:
$$ \Sigma = \frac{1}{2}
 \begin{bmatrix} 
0 & 1 & -1 \\ 
1 & 0 & -1 \\ 
0 & 0 & 0 \end{bmatrix}
\begin{bmatrix}
0 & 1 & 0   \\
1 & 0 & 0   \\
-1 & -1 & 0 \\
\end{bmatrix}=\frac{1}{2}
\begin{bmatrix}
2 & 1 & 0   \\
1 & 2 & 0   \\
0 & 0 & 0 \\
\end{bmatrix}$$

We should now compute the eigenvalues of the covariance matrix.


$$
det(\lambda-\Sigma) = \begin{bmatrix}
\lambda-1 & -\frac{1}{2} & 0   \\
-\frac{1}{2} & \lambda-1 & 0   \\
0 & 0 & \lambda \\
\end{bmatrix}=0
$$
which we can solve and get:
$$
\lambda(\lambda-\frac{1}{2})(\lambda-\frac{3}{2}) = 0
$$
So the eigenvalues of the covariant matrix are $\lambda_0=0$, $\lambda_1=\frac{1}{2}$ and $\lambda_2=\frac{3}{2}$.

Let us now calculate the eigenvectors of each eigenvalue. for $\lambda_2 = \frac{3}{2}$:
$$
\frac{1}{2}
\begin{bmatrix}
2 & 1 & 0   \\
1 & 2 & 0   \\
0 & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix} = \frac{3}{2}\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix} 
$$
So we get that,
$$
\begin{matrix}
a+\frac{1}{2}b=\frac{3}{2}a\\
\frac{1}{2}a+b=\frac{3}{2}b \\
0=c\\
\end{matrix}\implies a=b \text{ and } c=0 \text{ .}
$$
for $\lambda_1 = \frac{1}{2}$:
$$
\frac{1}{2}
\begin{bmatrix}
2 & 1 & 0   \\
1 & 2 & 0   \\
0 & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix} = \frac{1}{2}\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix} 
$$
So we get that,
$$
\begin{matrix}
a+\frac{1}{2}b=\frac{1}{2}a\\
\frac{1}{2}a+b=\frac{1}{2}b \\
0=c\\
\end{matrix}\implies a=-b \text{ and } c=0 \text{ .}
$$





for $\lambda_0 = 0$:
$$
\frac{1}{2}
\begin{bmatrix}
2 & 1 & 0   \\
1 & 2 & 0   \\
0 & 0 & 0 \\
\end{bmatrix}
\begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix} = 0 \begin{bmatrix}
a \\
b \\
c \\
\end{bmatrix} 
$$
So we get that,
$$
\begin{matrix}
a+\frac{1}{2}b=0\\
\frac{1}{2}a+b=0\\
0=0\\
\end{matrix}\implies a=-2 b \text{ and } c=1 \text{ .}
$$
So the eigenvectors for each eigenvalue are as follows, for $\lambda_2$ we get $v_2 = \begin{bmatrix}1 \\1 \\0 \\\end{bmatrix}$, for $\lambda_1$ we get $v_1 = \begin{bmatrix}1 \\-1 \\0 \\\end{bmatrix}$, and for $\lambda_0$ we get $v_0 = \begin{bmatrix}0 \\0 \\1 \\\end{bmatrix}$.
So the the first and second principal components of M are $\begin{bmatrix}1 \\1 \\0 \\\end{bmatrix}$ and $\begin{bmatrix}1 \\-1 \\0 \\\end{bmatrix}$, since $\lambda_2 > \lambda_1 >\lambda_0$.


## Question 3: K-means Clustering

### 1.
By randomizing the centers for every iterations until they converge into clusters, we avoid the problem of high dependence of the initial conditions. The randomization stops once the new centers are the same as the old centers.

### 2.
Solutions can be found in the file 'k_means.py'.
### 3.
The cost function decreases as K increases because we are essentially fitting the data in more clusters, at the limit we will put every single data point in it's own small cluster, this will decrease the cost function until there is essentially no error. This is over fitting, because having only single point clusters does not have any statistical significance. 
