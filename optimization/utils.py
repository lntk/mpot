import numpy as np


def ro(a, b):
    return np.sum(b - a) + np.sum(a * np.log(a / b))


def generate_cost_sequence(C, version):
    m = len(C.shape)
    n = C.shape[0]

    if version == 'v1':
        return list(range(m))  # A = [0, 1, ..., m - 1]

    if version == 'v2':
        C_max = np.amax(np.abs(C))
        if m == 2:
            return [0., 1.]
        else:
            return np.array([C_max + 1 - np.exp(i) for i in np.linspace(start=1, stop=np.log(C_max + 1), num=m)][1:]).tolist() + [1]

    return NotImplementedError


def extend_marginals(list_r, s, version='v1'):
    m = len(list_r)
    n = list_r[0].shape[0]

    Sigma_r = np.sum([np.sum(r) for r in list_r])

    if version == 'v1':
        return [np.append(r, np.array([[1 / (m - 1) * Sigma_r - np.sum(r) - 1 / (m - 1) * s]]), axis=0) for r in list_r]

    if version == 'v2':
        return [np.append(r, np.array([[Sigma_r - np.sum(r) - (m - 1) * s]]), axis=0) for r in list_r]

    raise NotImplementedError


def compute_C(x, y, z):
    n = x.shape[0]

    Cx = x.reshape(n, 1, 1)
    Cy = y.reshape(1, n, 1)
    Cz = z.reshape(1, 1, n)

    return 3 * (Cx ** 2 + Cy ** 2 + Cz ** 2) - (Cx + Cy + Cz) ** 2


def compute_B(beta, C, eta):
    m = 3
    n = beta.shape[1]
    C1 = np.ones(shape=(n, n, n)) * beta[0, :].reshape(n, 1, 1)
    C2 = np.ones(shape=(n, n, n)) * beta[1, :].reshape(1, n, 1)
    C3 = np.ones(shape=(n, n, n)) * beta[2, :].reshape(1, 1, n)
    return np.exp(C1 + C2 + C3 - C / eta)


def compute_E(beta, C, eta, weights):
    m = beta.shape[0]
    n = beta.shape[1]
    e = 0
    b = compute_B(beta, C, eta)
    for i in range(m):
        e += np.sum(np.abs(np.sum(b, axis=tuple([k for k in range(m) if k != i])) - weights[i]))
    return e


def extract_tensor(X, n, m):
    if m == 3:
        return X[:n, :n, :n]
    else:
        raise NotImplementedError
