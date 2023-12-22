import numpy as np


def get_dominant_eigenvalue_and_eigenvector(data, num_steps):
    """
    data: np.ndarray – symmetric diagonalizable real-valued matrix
    num_steps: int – number of power method steps

    Returns:
    eigenvalue: float – dominant eigenvalue estimation after `num_steps` steps
    eigenvector: np.ndarray – corresponding eigenvector estimation
    """
    ### YOUR CODE HERE
    b0 = np.ones(np.shape(data)[1]).T

    for i in range(num_steps):
        b1 = data.dot(b0)
        b1_norm = np.linalg.norm(data.dot(b0))
        b0 = b1 / b1_norm

    eigenvalue = b0.T.dot(data.dot(b0)) / (b0.T.dot(b0))

    return np.max(eigenvalue).item(), b0
