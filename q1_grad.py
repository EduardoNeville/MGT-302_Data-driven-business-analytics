import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def loss(x, y, theta):
    """
    # Loss is calculated here 
    """
    sum = 0
    h = np.dot(theta.T, x)
    for x_i, y_i in zip(x, y):
        sum += (h - y_i)**2 
    return np.divide(sum, 2)

def gradient_descent(X, y, theta, alpha, num_iterations):
    old_theta = theta
    num_samples, num_features = X.shape
    J_hist = np.zeros((num_iterations, 1))
    for i in range(num_iterations):
        gradient = loss(X, y, theta) 
        print(gradient)
        new_theta = old_theta - (alpha * gradient)
        if np.linalg.norm(new_theta - old_theta) < 1e-5:
            return new_theta, J_hist
        old_theta = new_theta
        J_hist[i] = new_theta
    return new_theta, J_hist


def plot_loss(J_hist):
    plt.plot(J_hist[0], label='alpha = 1e-2')
    #plt.plot(J_hist[1], label='alpha = 1e-3')
    #plt.plot(J_hist[2], label='alpha = 1e-4')
    plt.xlabel('Number of iterations')
    plt.ylabel('Cost function')
    plt.legend()
    plt.show()

def main():

    mat = sio.loadmat('Q1-Dataset/weighttrain.mat')
    data = np.array(mat['x'])
    x = np.array(mat['x'])
    x = np.array(np.divide(x, np.linalg.norm(x)))
    y = np.array(mat['y'])

    theta = np.zeros(x.shape) 
    num_iters = 500

    alpha = 1e-4

    # add , J_hist_1
    theta_1, J_hist_1 = gradient_descent(x, y, theta, alpha, num_iters)
    #plot_loss(J_hist_1)

    """
    alpha = 1e-3
    theta_2, J_hist_2 = train(x, y, theta, alpha, num_iters)

    alpha = 1e-4
    theta_3, J_hist_3= train(x, y, theta, alpha, num_iters)

    plot_cost_func([J_hist_1, J_hist_2, J_hist_3], y)

    """

if __name__ == "__main__":
    main()

