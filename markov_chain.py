import numpy as np
import scipy as sci
import matplotlib.pyplot as plt

def norm_random_vec(n):
    vector = np.random.rand(1, n)[0]
    return vector/np.sum(vector)

def norm_random_matrix(n):
     matrix = np.random.rand(n, n)
     mat_sums = np.sum(matrix, axis=1)

     return matrix/mat_sums.reshape(n,1)

def markov_chain(n, N):
    p = norm_random_vec(n)
    P = norm_random_matrix(n)

    values, vectors = (np.linalg.eig(P.T))
    p_stationary = vectors[:, np.argmax(values)]
    p_stationary = p_stationary/np.sum(p_stationary)

    norms = [np.linalg.norm(p-p_stationary)]

    for i in range(N):
        p = np.matmul(P.T, p)
        norms.append(np.linalg.norm(p-p_stationary))

    plt.plot(norms)
    plt.savefig('markov_chain_{}_{}.png'.format(n, N))

norm_random_matrix(3)
markov_chain(5, 10)