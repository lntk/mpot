import numpy as np
from tqdm import tqdm
from optimization.utils import ro, compute_B, compute_E, generate_cost_sequence, extract_tensor

global_info = {
    'index_tensor': None,
    'count_nplus1': None,
}


class Extension:
    def __init__(self):
        self.index_tensor = None
        self.count_nplus1 = None

    def get_indices_T(self, i):
        return self.index_tensor[self.count_nplus1 == i]

    def extend_cost(self, C, version='v1'):
        m = len(C.shape)
        n = C.shape[0]

        self.index_tensor = np.array([np.unravel_index(i, ((n + 1,) * m)) for i in range(int(np.power(n + 1, m)))])
        self.count_nplus1 = np.array([np.where(v == n)[0].shape[0] for v in self.index_tensor])

        A = generate_cost_sequence(C, version=version)

        C_extended = np.zeros(shape=((n + 1,) * m))

        for v in self.get_indices_T(0):
            v_to_indices = tuple(v.tolist())
            C_extended[v_to_indices] = C[v_to_indices]
        for i in range(1, m + 1):
            for v in self.get_indices_T(i):
                v_to_indices = tuple(v.tolist())
                C_extended[v_to_indices] = A[i - 1]

        return C_extended

    @staticmethod
    def extend_marginals(list_r, s, version='v1'):
        m = len(list_r)
        n = list_r[0].shape[0]

        Sigma_r = np.sum([np.sum(r) for r in list_r])

        if version == 'v1':
            return [np.append(r, np.array([[1 / (m - 1) * Sigma_r - np.sum(r) - 1 / (m - 1) * s]]), axis=0) for r in list_r]

        if version == 'v2':
            return [np.append(r, np.array([[Sigma_r - np.sum(r) - (m - 1) * s]]), axis=0) for r in list_r]

        raise NotImplementedError


def sinkhorn_mpot(C, list_a, s, eta, epsilon, max_iter, version='v1', logging=False, verbose=False):
    n = list_a[0].shape[0]
    m = len(C.shape)

    C_extended = Extension().extend_cost(C, version=version)
    list_a_extended = Extension.extend_marginals(list_a, s, version=version)

    X_mot, logs = sinkhorn_mot(C_extended, list_a_extended, eta, epsilon, max_iter, verbose=verbose, logging=logging)
    X_mpot = extract_tensor(X_mot, n, m)

    return np.sum(X_mpot * C), X_mpot, logs


def sinkhorn_mot(C, list_a, eta, epsilon, max_iter=100, logging=False, verbose=False):
    X, logs = multi_sinkhorn(C, eta, list_a, epsilon, max_iter, verbose=verbose, logging=logging)
    X = rounding(X, list_a)

    return X, logs


def multi_sinkhorn(C, eta, weights, epsilon, max_iter=100, logging=True, verbose=False):
    weights = [weight.reshape(-1, ) for weight in weights]

    logs = {
        'values': list(),
        'iterations': list(),
    }
    m = len(weights)
    n = weights[0].shape[0]
    beta = np.zeros((m, n))
    i = 0
    gaps = []
    values = []
    for i in tqdm(range(max_iter)):
        gap = compute_E(beta, C, eta, weights)
        if verbose:
            print(gap)
        # if gap <= epsilon:
        #     break

        # print('Iteration: '+str(i))
        b = compute_B(beta, C, eta)
        K = np.argmax([ro(weights[i], np.sum(b, axis=tuple([j for j in range(m) if j != i]))) for i in range(m)])
        beta[K] = beta[K] + np.log(weights[K]) - np.log(np.sum(b, axis=tuple([j for j in range(m) if j != K])))
        # gaps.append(gap)
        # values.append(np.sum(C * b))
        if logging:
            X = compute_B(beta, C, eta)
            X = rounding(X, weights)

            logs['iterations'].append(i + 1)
            logs['values'].append(np.sum(extract_tensor(X, n, m) * extract_tensor(C, n, m)))

    return compute_B(beta, C, eta), logs  # ,gaps,values


def rounding(X, weights):
    weights = [weight.reshape(-1, ) for weight in weights]

    m = len(weights)
    n = weights[0].shape[0]
    for k in range(m):
        X = X.reshape(tuple([n] * m))
        rk = np.sum(X, axis=tuple([j for j in range(m) if j != k]))
        A = np.concatenate([np.ones((1, n)), weights[k] / rk.reshape(1, -1)], axis=0)
        z_k = np.min(A, axis=0)
        X = X.reshape(-1, )
        for j in range(n):
            for i in range(n ** m):
                u = np.unravel_index(i, tuple([n] * m))
                if u[k] == j:
                    X[i] = z_k[j] * X[i]
    err = np.zeros((m, n))
    X = X.reshape(tuple([n] * m))
    for k in range(m):
        err[k] = weights[k] - np.sum(X, axis=tuple([j for j in range(m) if j != k]))
    Y = np.zeros((n ** m,))
    X = X.reshape(-1, )
    for i in range(n ** m):
        u = np.unravel_index(i, tuple([n] * m))
        Y[i] = X[i] + np.prod(np.array([err[k][u[k]] for k in range(m)])) / (np.sum(np.abs(err[0])) ** (m - 1))
    return Y.reshape(tuple([n]) * m)

#
#
# def compute_B(beta, C, eta):
#     n = C.shape[0]
#     m = len(C.shape)
#     C1 = np.ones(shape=(n, n, n)) * beta[0, :].reshape(n, 1, 1)
#     C2 = np.ones(shape=(n, n, n)) * beta[1, :].reshape(1, n, 1)
#     C3 = np.ones(shape=(n, n, n)) * beta[2, :].reshape(1, 1, n)
#     return np.exp(C1 + C2 + C3 - C / eta)
#
#
# def project_on_marginal(X, k, m):
#     return np.sum(X, axis=tuple([i for i in range(m) if i != k]))
#
#
# def compute_E(beta, C, eta, weights):
#     m = len(C.shape)
#     B = compute_B(beta, C, eta)
#
#     return sum([np.sum(np.abs(project_on_marginal(B, k, m) - weights[k])) for k in range(m)])


# def KL(a, b):
#     return np.sum(b - a) + np.sum(a * np.log(a / b))
#
#
# def rounding(X, weights):
#     n = X.shape[0]
#     m = len(X.shape)
#
#     for k in range(m):
#         X = X.reshape(tuple([n] * m))
#         rk = np.sum(X, axis=tuple([j for j in range(m) if j != k]))
#         A = np.concatenate([np.ones((1, n)), weights[k] / rk.reshape(1, -1)], axis=0)
#         z_k = np.min(A, axis=0)
#         X = X.reshape(-1, )
#         for j in range(n):
#             for i in range(n ** m):
#                 u = np.unravel_index(i, tuple([n] * m))
#                 if u[k] == j:
#                     X[i] = z_k[j] * X[i]
#     err = np.zeros((m, n))
#     X = X.reshape(tuple([n] * m))
#     for k in range(m):
#         err[k] = weights[k] - np.sum(X, axis=tuple([j for j in range(m) if j != k]))
#     Y = np.zeros((n ** m,))
#     X = X.reshape(-1, )
#     for i in range(n ** m):
#         u = np.unravel_index(i, tuple([n] * m))
#         Y[i] = X[i] + np.prod(np.array([err[k][u[k]] for k in range(m)])) / (np.sum(np.abs(err[0])) ** (m - 1))
#     return Y.reshape(tuple([n]) * m)
