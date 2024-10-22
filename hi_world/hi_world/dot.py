import numpy as np


def dot_product(a, b):
    """
    Compute the dot product of two 1D arrays.

    Parameters
    ----------
    a : numpy.ndarray
        First input array. Must be 1D.
    b : numpy.ndarray
        Second input array. Must be 1D and of the same length as `a`.

    Returns
    -------
    float
        The dot product of the two arrays.

    Raises
    ------
    ValueError
        If the input arrays are not of the same length.

    Examples
    --------
    >>> dot_product(np.array([1, 2, 3]), np.array([4, 5, 6]))
    32.0

    >>> dot_product(np.array([1, 0]), np.array([0, 1]))
    0.0
    """
    if a.shape[0] != b.shape[0]:
        raise ValueError("Input arrays must have the same length.")
    return np.dot(a, b)
