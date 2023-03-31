import numpy as np
from scipy.stats import multivariate_normal

# Define the Gaussian HMM model parameters
num_states = 2
num_obs = 2
A = np.array([[0.6, 0.4], [0.3, 0.7]])  # transition matrix
mean = np.array([[0.0, 0.0], [1.0, 1.0]])  # mean matrix
cov = np.array([[[1.0, 0.0], [0.0, 1.0]], [[1.0, 0.0], [0.0, 1.0]]])  # covariance matrix
pi = np.array([0.5, 0.5])  # initial state distribution
tol = 1e-6  # tolerance for convergence
max_iter = 1000  # maximum number of iterations

# Generate some example observations
np.random.seed(123)
obs = np.random.multivariate_normal(mean=[[-1.0, -1.0], [1.0, 1.0]], cov=cov, size=50)

# Run the EM algorithm to estimate the model parameters
log_likelihood_old = -np.inf
for i in range(max_iter):
    # E-step: compute the forward and backward probabilities
    alpha = np.zeros((len(obs), num_states))
    beta = np.zeros((len(obs), num_states))
    gamma = np.zeros((len(obs), num_states))
    xi = np.zeros((len(obs), num_states, num_states))
    alpha[0] = pi * multivariate_normal.pdf(obs[0], mean=mean, cov=cov)
    for t in range(1, len(obs)):
        alpha[t] = multivariate_normal.pdf(obs[t], mean=mean, cov=cov) * np.dot(alpha[t - 1], A)
    beta[-1] = 1.0
    for t in range(len(obs) - 2, -1, -1):
        beta[t] = np.dot(A, multivariate_normal.pdf(obs[t + 1], mean=mean, cov=cov) * beta[t + 1])
    gamma = alpha * beta / np.sum(alpha * beta, axis=1)[:, None]
    for t in range(len(obs) - 1):
        xi[t] = A * np.outer(alpha[t], beta[t + 1]) * multivariate_normal.pdf(obs[t + 1], mean=mean, cov=cov) / np.sum(alpha * beta, axis=1)[:, None]

    # M-step: update the model parameters
    pi = gamma[0]
    A = np.sum(xi, axis=0) / np.sum(xi, axis=(0, 1))[:, None]
    for j in range(num_states):
        mean[j] = np.dot(gamma[:, j], obs) / np.sum(gamma[:, j])
        cov[j] = np.dot(gamma[:, j] * (obs - mean[j]).T, (obs - mean[j])) / np.sum(gamma[:, j])
    log_likelihood_new = np.sum(np.log(np.sum(alpha * beta, axis=1)))
    
    # Check for convergence
    if log_likelihood_new - log_likelihood_old < tol:
        break
    log_likelihood_old = log_likelihood_new

# Print the estimated model parameters
print("Estimated transition matrix:")
print(A)
print("Estimated mean matrix:")
print(mean)
print("Estimated covariance")
