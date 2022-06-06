import numpy as np
from scipy.stats import multivariate_normal

def calculate_responsibilities(X, mean, sigma, pi, N, K):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param mean: means of K D-dimensional Gaussians, shape: (K, D)
    :param sigma: covariance matrices for K D-dimensional Gaussians, shape: (K, D, D)
    :param pi: component weights (weights for each Gaussian component), an array, shape (K, )
    :param N: number of data points
    :param K: number of clusters
    :return: responsibilities - Equation (5) from the HW4 sheet
    """
    responsibilities = np.zeros((N, K)) # gamma_nk from the HW sheet

    # TODO: calculate responsibilities gamma_nk
    # Do not forget to do it for each of K components 
    # To calculate the term N(x_n | mu_k, Sigma_k) use multivariate_normal.pdf function
    for n in range(N):
        arr = []
        for k in range(K):
            x = pi[k] * multivariate_normal.pdf(X[n], mean[k], sigma[k])
            arr.append(x)
        y = np.sum(arr, axis = 0)

        for a in arr:
            responsibilities[n, arr.index(a)] = a/y
        
    return responsibilities                                             


def update_parameters(X, mean, sigma, pi, responsibilities, N, K):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param mean: means of K D-dimensional Gaussians, shape: (K, D)
    :param sigma: covariance matrices for K D-dimensional Gaussians, shape: (K, D, D)
    :param pi: component weights (weights for each Gaussian component), an array, shape (K, )
    :param responsibilities: responsibilities for each data point n and cluster k: shape (N, K)
    :param N: number of data points
    :param K: number of clusters
    :return: mean_new - Equation (7) from the HW4 sheet, shape: (K, D),
             sigma_new - Equation (8) from the HW4 sheet, shape: (K, D, D),
             pi_new - Equation (9) from the HW4 sheet, an array: shape (K, )
    """

    mean_new = np.zeros_like(mean)
    sigma_new  = np.zeros_like(sigma)
    pi_new = np.zeros_like(pi)
    
    N_k = np.sum(responsibilities, axis=0)
    
    # Sigma is already calculated. Your task is to calculate mean_new and pi_new.
    # If you want, you can make this piece of code more efficient (optional),
    # and no points will be achieved only for rewriting it.
    # The points for this task will be given for implementing the equations.
    for k in range(K):
        gamma_nk = responsibilities[:, k].T

        # TODO: mean_new
        m = np.zeros_like(mean_new[k])
        for n in range(N):
            m += gamma_nk[n] * X[n]

        mean_new[k] = m/N_k[k] 

        #sigma_new
        tmp = np.zeros_like(sigma_new[k])
        for sample in range(N):
            diff = (X[sample, :] - mean_new[k, :]).reshape((-1, 1))
            tmp += gamma_nk[sample] * np.dot(diff, diff.T)
                            
        sigma_new[k] = tmp / N_k[k]

        # TODO: pi_new
        pi_new[k] = N_k[k]/N
    
    return mean_new, sigma_new, pi_new


def em(X, K, max_iter):
    """
    :param X: data for clustering, shape: (N, D), with N - number of data points, D - dimension
    :param K: number of clusters
    :param max_iter:
    :return: mean - means of K D-dimensional Gaussians, shape: (K, D)
            soft_clusters - soft assignment of data points to clusters, shape: (N, K)
            log_likelihood - an array with values of cost over iteration
    """
    N = X.shape[0]
    D = X.shape[1]

    eps = 0.01

    # Init GMM
    init_variance = 1.5
    mean = np.random.random(size=(K, D))
     
    cov_mat = np.eye(D) * init_variance
    sigma = np.repeat(cov_mat[np.newaxis, :, :], K, axis=0)

    pi = 1. / K * np.ones((K,))
    assert np.isclose(np.sum(pi), 1.0), "The sum over Pi must equal to 1!"

    print(f'Init mean: {mean}')
    print(f'Init sigma: {sigma}')
    print(f'Init pi: {pi}')

    log_likelihood = []

    for it in range(max_iter):
        # E-Step
        # TODO: appropriate function call
        responsibilities = calculate_responsibilities(X, mean, sigma, pi, N, K)
        
        # M-Step
        # TODO: appropriate function call
        mean, sigma, pi = update_parameters(X, mean, sigma, pi, responsibilities, N, K)
        
        # Evaluate
        soft_clusters = np.zeros((N, K))
        for k in range(K):
            soft_clusters[:, k] = pi[k] * multivariate_normal.pdf(X, mean=mean[k, :], cov=sigma[k])
            
        log_likelihood.append(np.sum(np.log(np.sum(soft_clusters, axis=1))))

        if it > 1 and np.abs(log_likelihood[-1] - log_likelihood[-2]) < eps:
            print(f'Iteration {it}. Algorithm converged.')
            break
    print(f'Mean: {mean}')
    print(f'Sigma: {sigma}')
    print(f'Pi: {pi}')

    return mean, soft_clusters, log_likelihood
