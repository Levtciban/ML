import numpy as np


def prod_non_zero_diag(x):
    """Compute product of nonzero elements from matrix diagonal.

    input:
    x -- 2-d numpy array
    output:
    product -- integer number


    Vectorized implementation.
    """
    return np.prod(np.diag(x)[np.diag(x) != 0])


def are_multisets_equal(x, y):
    """Return True if both vectors create equal multisets.

    input:
    x, y -- 1-d numpy arrays
    output:
    True if multisets are equal, False otherwise -- boolean

    Vectorized implementation.
    """

    return np.array_equal(np.sort(x), np.sort(y))


def max_after_zero(x):
    """Find max element after zero in array.

    input:
    x -- 1-d numpy array
    output:
    maximum element after zero -- integer number

    Vectorized implementation.
    """

    ind = np.array(np.where(x == 0)) + 1
    return np.max(x[ind])



def convert_image(img, coefs):
    """Sum up image channels with weights from coefs array

    input:
    img -- 3-d numpy array (H x W x 3)
    coefs -- 1-d numpy array (length 3)
    output:
    img -- 2-d numpy array

    Vectorized implementation.
    """

    return np.dot(img, coefs)


def run_length_encoding(x):
    """Make run-length encoding.

    input:
    x -- 1-d numpy array
    output:
    elements, counters -- integer iterables

    Vectorized implementation.
    """
    r1 = np.insert(x, 0, -10 ** 2)
    r1 = r1[:-1]
    y = x - r1
    ind = np.where(y != 0)
    ty = np.append(ind, len(x))
    ty = ty[1:]
    return x[ind], ty - ind 



def pairwise_distance(x, y):
    """Return pairwise object distance.

    input:
    x, y -- 2d numpy arrays
    output:
    distance array -- 2d numpy array

    Vctorized implementation.
    """
    xsq = np.sum(x**2, axis=1, keepdims=True)
    ysq = np.sum(y**2, axis=1)
    xys = np.dot(x, y.T)
    
    squared_distances = xsq - 2 * xys + ysq
    squared_distances = np.maximum(squared_distances, 0)
    
    return np.sqrt(squared_distances)
