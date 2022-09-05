# https://github.com/freelunchtheorem/Conditional_Density_Estimation
import numpy as np

def norm_along_axis_1(A, B, squared=False, norm_dim=False):
    """ calculates the (squared) euclidean distance along the axis 1 of both 2d arrays

    Args:
      A: numpy array of shape (n, k)
      B: numpy array of shape (m, k)
      squared: boolean that indicates whether the squared euclidean distance shall be returned, \
               otherwise the euclidean distance is returned
      norm_dim: (boolean) normalized the distance by the dimensionality k -> divides result by sqrt(k)

      Returns:
         euclidean distance along the axis 1 of both 2d arrays - numpy array of shape (n, m)
    """
    assert A.shape[1] == B.shape[1]
    result = np.zeros(shape=(A.shape[0], B.shape[0]))

    if squared:
        for i in range(B.shape[0]):
            result[:, i] = np.sum(np.square(A - B[i, :]), axis=1)
    else:
        for i in range(B.shape[0]):
            result[:, i] = np.linalg.norm(A - B[i, :], axis=1)

    if norm_dim:
        result = result / np.sqrt(A.shape[1])
    return result
