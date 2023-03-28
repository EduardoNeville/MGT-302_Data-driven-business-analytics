import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def init_centers(data, K):
    np.random.seed(0)
    random_idx = np.random.permutation(data.shape[0])
    centers = data[random_idx[:K]]
    return centers

def compute_distance(data, centers, K):
    distance = np.zeros((data.shape[0], K))
    for i in range(K):
        distance[:, i] = np.sum((data - centers[i]) ** 2, axis=1)
    return distance

def find_closest_cluster(distance):
    return np.argmin(distance, axis=1)

def compute_centers(data, cluster_assignments, K):
    centers = np.zeros((K, data.shape[1]))
    for i in range(K):
        centers[i, :] = np.mean(data[cluster_assignments == i], axis=0)
    return centers

def K_means(data, K, max_iter):
    centers = init_centers(data, K)
    for i in range(max_iter):
        old_centers = centers.copy()
        distance = compute_distance(data, old_centers, K)
        cluster_assignments = find_closest_cluster(distance)
        centers = compute_centers(data, cluster_assignments, K)

        if np.all(old_centers == centers):
            break
    return centers, cluster_assignments

# 5. Plot the data points and the cluster centroids
def plot_data(data, centers, cluster_assignments):
    plt.scatter(data[:,0], data[:,1], c=cluster_assignments, cmap='rainbow')
    plt.scatter(centers[:,0], centers[:,1], c='black')
    plt.show()

def elbow_method(data, centers, cluster_assignments, K):
    J = 0
    for i in range(K):
        J += np.sum((data[cluster_assignments == i] - centers[i]) ** 2)
    return J


def main():
    mat = sio.loadmat('../Q2-Dataset/kmeans.mat')
    data = mat['kmeans']
    K = 10

    #plot_data(data, centers, cluster_assignments)

    J_s = np.zeros(K)
    for i in range(1, K):
        centers, cluster_assignments = K_means(data, i, 100)
        J_s[i] = elbow_method(data, centers, cluster_assignments, i)

    plt.plot(range(1, K+1), J_s)
    plt.xlabel('Number of clusters')
    plt.ylabel('Loss Function')
    plt.show()

if __name__ == "__main__":
    main()
