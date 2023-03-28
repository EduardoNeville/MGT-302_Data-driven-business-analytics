import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt

def k_means(data, numb_centers):

    # 1. Randomly initialize K cluster centroids
    centers = np.ones((numb_centers, data.shape[1]))
    for i in range(0, numb_centers):
        centers[i] = data[np.random.randint(0,data.shape[1])]
    print(centers)

    # 2. Assign each data point to the closest cluster centroid
    idx_center = np.zeros(data.shape[0])
    for i in range(data.shape[0]):
        min_dist = 10000
        for j in range(numb_centers):
            dist = np.linalg.norm(data[i] - centers[j])
            if dist < min_dist:
                min_dist = dist
                min_index = j
                idx_center[i] = min_index

    data = np.c_[data, idx_center]
    # 3. Update the cluster centroids to be the mean of the data points assigned to it
    for i in range(numb_centers):
        if len(data[data[:,2] == i]) > 0:
            centers[i] = np.mean(data[data[:,2] == i], axis=0)[:2]
    return centers, data

def cost_function(data, centers):
    # 5. Compute the cost function
    cost = 0
    for i in range(len(centers)):
        for j in range(data.shape[0]):
            cost += np.linalg.norm(data[j,:2] - centers[i])
    return cost

# 5. Plot the data points and the cluster centroids
def plot_data(data, centers):
    plt.scatter(data[:,0], data[:,1], c=data[:,2])
    plt.scatter(centers[:,0], centers[:,1], c='red')
    plt.show()

def main():
    mat = sio.loadmat("../Q2-Dataset/kmeans.mat")

    data = np.array(mat['kmeans'])
    numb_centers = 4

    numb_iter = 100
    for i in range(numb_iter):
        centers, data = k_means(data, numb_centers)
        print(f"Iteration {i+1} \n Centers: \n {centers} \n ")
    plot_data(data, centers)

if __name__ == "__main__":
    main()
