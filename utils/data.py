import numpy as np


def init_cost(size, bound_min=1, bound_max=50, symmetric=True, max_value=None, zero_diagonal=False):
    # if len(size) > 2:
    #     raise NotImplementedError

    if max_value is not None:
        C = np.random.uniform(low=bound_min, high=bound_max, size=size)
        C_max = np.amax(C)
        C = C / C_max * max_value
    else:
        C = np.random.uniform(low=bound_min, high=bound_max, size=size)

    if symmetric:
        C = (C + C.T) / 2

    if zero_diagonal:
        for i in range(C.shape[0]):
            C[i, i] = 0.

    return C.astype("float64")


def init_measure(n, alpha, min_value=None, max_value=None, tolerance=100):
    if min_value is not None and max_value is not None:
        raise NotImplementedError

    if max_value is not None:
        count = 0
        while True:
            alpha_minus = alpha - max_value
            a = np.random.uniform(low=0.1, high=1, size=(n - 1, 1))
            a = a / a.sum() * alpha_minus

            if np.amax(a) < max_value:
                a = np.concatenate([a, np.array([max_value]).reshape(-1, 1)], axis=0)
                np.random.shuffle(a)
                break

            count += 1
            if count > tolerance:
                raise Exception("Cannot init measure.")
    elif min_value is not None:
        count = 0
        while True:
            alpha_minus = alpha - min_value
            a = np.random.uniform(low=0.1, high=1, size=(n - 1, 1))
            a = a / a.sum() * alpha_minus

            if np.amin(a) > min_value:
                a = np.concatenate([a, np.array([min_value]).reshape(-1, 1)], axis=0)
                np.random.shuffle(a)
                break

            count += 1
            if count > tolerance:
                raise Exception("Cannot init measure.")
    else:
        a = np.random.uniform(low=0.1, high=1, size=(n, 1))
        a = a / a.sum() * alpha

    return a.astype("float64")


def init_data(shape, scale=1., offset=0., distribution="gaussian"):
    if distribution == "gaussian":
        X = np.random.normal(loc=5., scale=5., size=shape)
    elif distribution == "uniform":
        X = (np.random.rand(*shape) - 0.5) * scale + offset
    else:
        raise NotImplementedError

    return X.astype("float64")
