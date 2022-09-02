import numpy as np

def execute_batch_pdf(pdf_fun, X, Y, batch_size=None):
    """
    Executes pdf_fun in batches in multiple processes and concatenates results along axis 0

    Args:
        pdf_fun: callable with signature pdf(X, Y) returning a numpy array
        X: ndarray with shape (n_queries, ndim_x)
        Y: ndarray with shape (n_queries, ndim_y)
        batch_size: (optional) integer denoting the batch size for the individual function calls

    Returns:
        ndarray of shape (n_queries,) which results from a concatenation of all pdf calls
    """
    if batch_size is None:
        n_batches = n_jobs
    else:
        n_batches = query_length // batch_size + int(not (query_length % batch_size == 0))

    X_batches, Y_batches, indices = _split_into_batches(X, Y, n_batches)

    # TODO
    result = []
    for X_batch, Y_batch in zip(X_batches, Y_batches):
        result.extend(pdf_fun(X, Y))

    return result

def _split_into_batches(X, Y, n_batches):
    assert X.shape[0] == X.shape[0]
    if n_batches <= 1:
        return [X], [Y], range(1)
    else:
        return np.array_split(X, n_batches, axis=0), np.array_split(Y, n_batches, axis=0), range(n_batches)
