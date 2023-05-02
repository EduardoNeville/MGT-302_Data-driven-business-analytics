import numpy as np
import matplotlib.pyplot as plt
import scipy.io as sio

def init_centers(data, K):
    np.random.seed(0)
    random_idx = np.random.permutation(data.shape[0])
    centers = data[random_idx[:K]]
    return centers

def compute_distance(pt, centers, K):
    """
        Compute distance from a 2d data point to a 2d center
    """
    distances = np.zeros((pt.shape[0], K))
    for i in range(K):
        distances[:, i] = np.linalg.norm(pt - centers[i], axis=1)
    return distances

def compute_centers(data, cluster_assignments, K):
    centers = np.zeros((K, data.shape[1]))
    for i in range(K):
        centers[i, :] = np.mean(data[cluster_assignments == i], axis=0)
    return centers

def K_means(data, K, max_iter):
    centers = init_centers(data, K)
    cluster_assignments = np.zeros(data.shape[0])
    for i in range(max_iter):
        old_centers = centers.copy()
        for j in range(data.shape[0]):
            distances = compute_distance(data, old_centers, K)
            cluster_assignments[j] = np.argmin(distances[j]) 
        centers = compute_centers(data, cluster_assignments, K)

        if np.all(old_centers == centers):
            break
    return centers, cluster_assignments

# 5. Plot the data points and the cluster centroids
def plt_dt(data, centers, cluster_assignments, k):
    colors = ['r', 'g', 'b', 'y']
    fig, ax = plt.subplots()
    for i in range(k):
        points = np.array([data[j] for j in range(data.shape[0]) if cluster_assignments[j] == i])
        ax.scatter(points[:, 0], points[:, 1], s=7, c=colors[i])
    ax.scatter(centers[:, 0], centers[:, 1], marker='*', s=200, c='#050505')
    plt.show()

def elbow_method(data, centers, cluster_assignments, K):
    J = 0
    for i in range(K):
        J += np.sum((data[cluster_assignments == i] - centers[i]) ** 2)
    return J


def main():
    mat = sio.loadmat('Q2-Dataset/kmeans.mat')
    data = mat['kmeans']
#    print(data[:])

    # k means with 4 clusters
    centers, cluster_assignments = K_means(data, 4, 100)
    plt_dt(data, centers, cluster_assignments, 4)

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
