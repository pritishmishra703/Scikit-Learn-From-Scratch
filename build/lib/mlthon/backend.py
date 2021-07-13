import numpy as np
import warnings

def _check_data_validity(data, names):
    '''
    Checks if data is valid for training (or any other operation).

    Parameters
    ----------
    data: list
        A list of data which is to be checked for validity.

    names: list
        Names of the data. Which should be consecutive to data list.

    Example
    -------
    >>> from mlthon.backend import _check_data_validity
    >>> X, y = np.array([1., 2., 3.]), np.array([2., 3., 4.])
    >>> _check_data_validity(data=[X, y], names=['X', 'y'])
    '''
    
    for datum, name in zip(data, names):
        if not isinstance(datum, np.ndarray):
            warning = f"{name} should be of type 'np.ndarray'"
            raise TypeError(warning)

def _dim_check(data, dim, name):
    if len(data.shape) != dim:
        raise ValueError(f"Dimension of {name} should be strictly equal to {dim}")
