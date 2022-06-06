import numpy as np

def euclidean_distance(x, y):
    """
    :param x: D-dimensional vector
    :param y: D-dimensional vector
    :return: dist - scalar value
    """
    dist = np.linalg.norm(x - y) # TODO: implement 
    #print(x, y, dist)
    return dist

def cost_function(X, K, ind_samples_clusters, centroids):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: cost - a scalar value
    """
    J = 0
    N = X.shape[0]

    # TODO: implement
    for n in range(N):
        for k in range(K):
            temp = ind_samples_clusters[n, k] * np.square(euclidean_distance(X[n], centroids[k]))
            J += temp

    return J

def closest_centroid(sample, centroids):
    """
    :param sample: a data point x_n (of dimension D)
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: idx_closest_cluster - index of the closest cluster
    """
    # Calculate distance of the current sample to each centroid
    # Return the index of the closest centroid (int value from 0 to (K-1))
    K = centroids.shape[0]
    
    distances = [] # TODO: change
    idx_closest_cluster = 0 # TODO: change

    for k in range(K):
        distances.append(np.square(euclidean_distance(sample, centroids[k])))

    idx_closest_cluster = np.argmin(distances)

    return idx_closest_cluster

def assign_samples_to_clusters(X, K, centroids):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param centroids: means of clusters, K vectors of dimension D, shape: (K, D)
    :return: ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    """

    N = X.shape[0] # N - number of samples

    ind_samples_clusters = np.zeros((N, K))

    # TODO: implement
    # There will be a function call to closest_centroid function
    for n in range(N):
        index = closest_centroid(X[n], centroids)
        ind_samples_clusters[n, index] = 1

    assert np.min(ind_samples_clusters) == 0 and np.max(ind_samples_clusters == 1), "These must be one-hot vectors"
    return ind_samples_clusters

def recompute_centroids(X, K, ind_samples_clusters):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param ind_samples_clusters: indicator variables for all data points, shape: (N, K)
    :return: centroids - means of clusters, shape: (K, D)
    """
    N = X.shape[0]
    D = X.shape[1]
    
    centroids = np.zeros((K, D))
    
    # TODO: Implement the equation
    for k in range(K):
        arr = []
        sum2 = 0
        for n in range(N):
            if ind_samples_clusters[n, k] == 1:
                sum2 += 1
                arr.append(ind_samples_clusters[n, k] * X[n])

        sum1 = np.sum(arr, axis = 0)
        centroids[k] = sum1/sum2

    return centroids

def kmeans(X, K, max_iter):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter:
    :return: ind_samples_clusters - indicator variables for all data points, shape: (N, K)
            centroids - means of clusters, shape: (K, D)
            cost - an array with values of cost over iteration
    """

    n, d = X.shape

    # Init centroids
    rnd_points = np.random.randint(low=0, high=n, size=K)
    centroids = X[rnd_points, :]
    eps = 1e-6

    print(f'Init centroids: {centroids}')

    cost = []
    for it in range(max_iter):    

        # Assign samples to the clusters
        ind_samples_clusters = assign_samples_to_clusters(X, K, centroids) # TODO: function call
        J = cost_function(X, K, ind_samples_clusters, centroids) # TODO: function call to evaluate cost
        cost.append(J)
        
        # Calculate new centroids from the clusters
        centroids = recompute_centroids(X, K, ind_samples_clusters) # TODO: function call
        J = cost_function(X, K, ind_samples_clusters, centroids)  # TODO: function call to evaluate cost again
        cost.append(J)
        
        if it > 0 and np.abs(cost[-1] - cost[-2]) < eps:
            print(f'Iteration {it+1}. Algorithm converged.')
            print(f'New centroids: {centroids}')
            break

    return ind_samples_clusters, centroids, cost

