import numpy as np
import scipy as sci 
import matplotlib.pyplot as plt

def norm_random_vec(n):
    '''
    Returns a normalized random n-vector.
    '''
    vector = np.random.rand(1, n)[0]
    return vector/np.sum(vector)

def norm_random_matrix(n):
    '''
    Returns a normalized random nxn matrix.
    '''
    matrix = np.random.rand(n, n)
    mat_sums = np.sum(matrix, axis=1)

    return matrix/mat_sums.T

def markov_chain(n, N):
    '''
    Simulating N steps of a Markov chain with n states. 

    This function starts begins with a random probability distribution
    between states, and transitions between states according to a 
    randomly generated transition matrix.
    '''

    p = norm_random_vec(n)      # random intial probability distribution
    P = norm_random_matrix(n)   # random transition matrix

    # computing the eigenvector with the largest associated eigenvalue
    # lambda = 1
    values, vectors = (np.linalg.eig(P.T))
    p_stationary = vectors[:, np.argmax(values)]
    p_stationary = p_stationary/np.sum(p_stationary)

    norms = [np.linalg.norm(p-p_stationary)]

    # stepping through N iterations of the Markov chain
    for i in range(N):
        p = np.matmul(P.T, p)   # updating current probability distribution between states
        # recording the normalized difference p and p_stationary
        # for each step
        norms.append(np.linalg.norm(p-p_stationary))

    # plotting the normalized difference between p and p_stationary
    # against the number of steps/iterations of the Markov chain
    plt.plot(norms)
    plt.savefig('markov_chain_{}_{}.png'.format(n, N))

# running function with n = 5, N = 10
markov_chain(5, 10)