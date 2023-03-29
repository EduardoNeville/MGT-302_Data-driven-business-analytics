import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def train(x, y, theta, alpha, num_iters):
    """
        Training the linear regression through a num_iterations
    """
    m = len(y)
    J_hist = np.zeros((num_iters, 1))
    for i in range(num_iters):
        new_theta = theta - alpha * cost_function(x,y,theta)            
        print(new_theta)
        # When the error term is too small we stop doing more iterations as we have found a minima
        if np.linalg.norm(new_theta - theta) < 1e-5:
            return theta, J_hist
        theta = new_theta
        J_hist[i] = theta
    return theta, J_hist

def cost_function(x, y, theta):
    """
        the cost function is calculated here to find the new theta
    """
    sum = 0
    for x_i, y_i in zip(x, y):
        h = np.dot(theta, np.linalg.norm(x_i))
        sum = sum + (h - np.linalg.norm(y_i)) * np.linalg.norm(x_i)
    return np.divide(sum, 2**np.size(x,0))


def plot_cost_func(J_hist, y):
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
    norm_x = np.linalg.norm(x)
    x = np.array(np.divide(x, norm_x))
    y = np.array(mat['y'])

    theta = np.ones((5,))
    num_iters = 500

    alpha = 1e-2
    theta_1, J_hist_1 = train(norm_x, y, theta, alpha, num_iters)
    print(f"theta_1: \n {theta_1}")
    plot_cost_func([J_hist_1], y)

    """
    alpha = 1e-3
    theta_2, J_hist_2 = train(x, y, theta, alpha, num_iters)

    alpha = 1e-4
    theta_3, J_hist_3= train(x, y, theta, alpha, num_iters)

    plot_cost_func([J_hist_1, J_hist_2, J_hist_3], y)

    """

if __name__ == "__main__":
    main()

