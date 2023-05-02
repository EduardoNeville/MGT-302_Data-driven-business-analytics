import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def loss_function(X, y, theta):
    """
        Stochastic gradient descent loss function
    """
    old_loss = 0.0
    for i in range(X.shape[0]):
        new_loss = old_loss - np.log(1 + np.exp(-y[i] * np.dot(X[i], theta)))
        if np.abs(new_loss - old_loss) < 1e-10:
            break 
        #loss += -y[i] * np.dot(X[i], w)))
    return new_loss

# initialize theta to zeros
"""
def fit(X, y, theta, epochs=100, batch_size=1):
    theta = np.zeros(X.shape[1])
    # perform stochastic gradient descent
    for epoch in range(epochs):
        # shuffle the data
        indices = np.random.permutation(X.shape[0])
        X = X[indices]
        y = y[indices]
        # iterate over batches
        for i in range(0, X.shape[0], batch_size):
            X_batch = X[i:i+batch_size]
            y_batch = y[i:i+batch_size]
            # compute the gradient for the batch
            y_hat = np.dot(X_batch, theta)
            error = y_hat - y_batch
            gradient = alpha * np.dot(X_batch.T, error) / batch_size
            # update the theta values
            theta -= gradient
    return theta
"""

def fit(X, y, num_iterations, batch_size, alpha):
    n_samples, n_features = X.shape
    theta = np.zeros(n_features)
    bias = 0
    losses = []
    for i in range(num_iterations):
        batch_indices = np.random.choice(n_samples, batch_size, replace=False)
        X_batch = X[batch_indices]
        y_batch = y[batch_indices]
        y_predicted = np.dot(X_batch, theta) + bias
        dw = (1 / batch_size) * np.dot(X_batch.T, (y_predicted - y_batch))
        db = (1 / batch_size) * np.sum(y_predicted - y_batch)
        theta -= alpha *  dw.squeeze()
        bias -= alpha * db
        loss = np.mean((y_predicted - y_batch) ** 2)
        losses.append(loss)
    return theta, bias, losses

def gradient_descent(X, y, theta, alpha, num_iter):
    """
        Stochastic gradient descent
    """
    loss = np.ones(num_iter)
    for i in range(num_iter):
        loss[i] = loss_function(X, y, theta)
        grad, theta = gradient(X, y, theta, alpha)
        print('Iteration: {}, Loss: {}'.format(i, loss[i]))
    return theta, loss

def plot_loss(loss):
    """
        Plot loss function
    """
    plt.plot(loss)
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.title('Loss function')
    plt.show()


def plot_loss_multiple_alpha(loss, alpha):
    """
        Plot loss function
    """
    plt.plot(loss[0], label='alpha = 1e-10')
    plt.plot(loss[1], label='alpha = 1e-5')
    plt.plot(loss[2], label='alpha = 1e-4')
    plt.plot(loss[3], label='alpha = 1e-3')
    plt.plot(loss[4], label='alpha = 1e-2')
    plt.legend()
    plt.xlabel('Number of iterations')
    plt.ylabel('Loss')
    plt.title('Loss function')
    plt.show()
    
def train(x, y, theta, numb_iter):
    """
        Train the model
        And ploting the different loss functions for different alpha's
    """
    alpha = [1e-10, 1e-5, 1e-4, 1e-3, 1e-2]
    loss = np.zeros((len(alpha), numb_iter))
    for i in range(len(alpha)):
        theta, loss[i] = gradient_descent(x, y, theta, alpha[i], numb_iter)
    return loss, alpha

def main():
    mat = sio.loadmat('Q1-Dataset/weighttrain.mat')
    data = np.array(mat['x']) 
    x = np.divide(mat['x'], np.linalg.norm(mat['x']))
    y = np.array(mat['y']) 

    theta = np.zeros(x.shape[1])
    # Making the first elt of theta 1
    theta[0] = 1
    alpha = 1e-4 
    numb_iter = 5000
    
    theta, bias, loss = fit(x, y, numb_iter, 1, alpha)
    #theta, loss = gradient_descent(x, y, theta, alpha, numb_iter)
    plot_loss(loss)

   # loss, alpha = train(x, y, theta, numb_iter)
   # plot_loss_multiple_alpha(loss, alpha)

if __name__ == '__main__':
    main()




