import numpy as np
import warnings

class BaseDensityEstimator:
    """ Interface for conditional density estimation models """

    def fit(self, X, Y, verbose=False):
        """ Fits the conditional density model with provided data

          Args:
            X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
            Y: numpy array of y targets - shape: (n_samples, n_dim_y)
        """
        raise NotImplementedError


    def pdf(self, X, Y):
        """ Predicts the conditional likelihood p(y|x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
              conditional likelihood p(y|x) - numpy array of shape (n_query_samples, )
        """
        raise NotImplementedError

    def log_pdf(self, X, Y):
        """ Predicts the conditional log-probability log p(y|x). Requires the model to be fitted.

           Args:
             X: numpy array to be conditioned on - shape: (n_samples, n_dim_x)
             Y: numpy array of y targets - shape: (n_samples, n_dim_y)

           Returns:
              conditional log-probability log p(y|x) - numpy array of shape (n_query_samples, )
         """
        # This method is numerically unfavorable and should be overwritten with a numerically stable method
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            log_prob = np.log(self.pdf(X, Y))
        return log_prob

    def score(self, X, Y):
        """Computes the mean conditional log-likelihood of the provided data (X, Y)

          Args:
            X: numpy array to be conditioned on - shape: (n_query_samples, n_dim_x)
            Y: numpy array of y targets - shape: (n_query_samples, n_dim_y)

          Returns:
            average log likelihood of data
        """
        return np.mean(self.log_pdf(X, Y))

    def _handle_input_dimensionality(self, X, Y=None, fitting=False):
        # assert that both X an Y are 2D arrays with shape (n_samples, n_dim)
        if X.ndim == 1:
            X = np.expand_dims(X, axis=1)

        if Y is not None:
            if Y.ndim == 1:
                Y = np.expand_dims(Y, axis=1)

            assert X.shape[0] == Y.shape[0], "X and Y must have the same length along axis 0"
            assert X.ndim == Y.ndim == 2, "X and Y must be matrices"

        if fitting:  # store n_dim of training data
            self.ndim_y, self.ndim_x = Y.shape[1], X.shape[1]
        else:
            assert X.shape[1] == self.ndim_x, f"X must have shape (?, {self.ndim_x}) but provided X has shape {X.shape}"
            if Y is not None:
                assert Y.shape[1] == self.ndim_y, f"Y must have shape (?, {self.ndim_y}) but provided Y has shape {Y.shape}"

        if Y is None:
            return X
        else:
            return X, Y
