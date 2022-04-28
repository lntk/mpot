import numpy as np
import cvxpy as cp


def flatten_indices_to_marginal_tensor_indices(shape):
    n = shape[0]
    m = len(shape)

    index_tensor = np.array([np.unravel_index(i, shape) for i in range(np.prod(shape))]).T

    mapping = [[np.where(index_tensor[i] == j)[0].tolist() for j in range(n)] for i in range(m)]
    return mapping


def solve_mpot(C, list_a, s, verbose=False):
    n = C.shape[0]
    m = len(C.shape)
    C_flatten = C.flatten()

    X = cp.Variable(C_flatten.shape)
    cost = cp.sum(cp.multiply(X, C_flatten))

    index_mapping = flatten_indices_to_marginal_tensor_indices(C.shape)
    rX = [[cp.sum(X[index_mapping[i][j]]) for j in range(n)] for i in range(m)]

    objective = cp.Minimize(cost)
    constraints = [0 <= X, cp.sum(X) == s] + [rX[i][j] <= list_a[i][j] for j in range(n) for i in range(m)]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK', verbose=verbose)

    return prob.value, X.value.reshape(C.shape)


def solve_entropic_mpot(C, list_a, s, eta, verbose=False):
    n = C.shape[0]
    m = len(C.shape)
    C_flatten = C.flatten()

    X = cp.Variable(C_flatten.shape)
    cost = cp.sum(cp.multiply(X, C_flatten))
    entropy_X = cp.sum(cp.entr(X)) + cp.sum(X)
    total_cost = cost - eta * entropy_X

    index_mapping = flatten_indices_to_marginal_tensor_indices(C.shape)
    rX = [[cp.sum(X[index_mapping[i][j]]) for j in range(n)] for i in range(m)]

    objective = cp.Minimize(total_cost)
    constraints = [0 <= X, cp.sum(X) == s] + [rX[i][j] <= list_a[i][j] for j in range(n) for i in range(m)]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK', verbose=verbose)

    return prob.value, X.value.reshape(C.shape)


def solve_mot(C, list_a, verbose=False):
    n = C.shape[0]
    m = len(C.shape)
    C_flatten = C.flatten()

    X = cp.Variable(C_flatten.shape)
    cost = cp.sum(cp.multiply(X, C_flatten))

    index_mapping = flatten_indices_to_marginal_tensor_indices(C.shape)
    rX = [[cp.sum(X[index_mapping[i][j]]) for j in range(n)] for i in range(m)]

    objective = cp.Minimize(cost)
    constraints = [0 <= X] + [rX[i][j] == list_a[i][j] for j in range(n) for i in range(m)]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver='MOSEK', verbose=verbose)

    return prob.value, X.value.reshape(C.shape)


def solve_entropic_ot_barycenter(list_omega, list_C, list_a, eta, verbose=False, return_status=False, solver="ECOS"):
    total_cost = cp.Constant(value=0)
    num_measure = len(list_omega)

    list_X = list()
    list_sum_X = list()
    list_sum_rowX = list()
    list_sum_colX = list()

    for i in range(num_measure):
        C = list_C[i]

        X = cp.Variable(C.shape)

        sum_X = cp.sum(X)
        sum_rowX = cp.sum(X, axis=1)
        sum_colX = cp.sum(X, axis=0)

        cost = cp.sum(cp.multiply(X, C))
        entropy_X = cp.sum(cp.entr(X)) + cp.sum(X)
        total_cost += list_omega[i] * (cost - eta * entropy_X)

        list_X.append(X)
        list_sum_X.append(sum_X)
        list_sum_rowX.append(sum_rowX)
        list_sum_colX.append(sum_colX)

    objective = cp.Minimize(total_cost)

    constraints = list()
    for i in range(num_measure):
        if i < num_measure - 1:
            constraints += [0 <= list_X[i], list_sum_rowX[i] - list_a[i].reshape(-1, ) == 0, list_sum_colX[i] - list_sum_colX[i + 1] == 0]
        else:
            constraints += [0 <= list_X[i], list_sum_rowX[i] - list_a[i].reshape(-1, ) == 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if return_status:
        return prob.value, [X.value for X in list_X], prob.status

    return prob.value, [X.value for X in list_X]


def solve_entropic_pot_barycenter(list_omega, list_C, list_a, s, eta, verbose=False, return_status=False, solver="ECOS"):
    total_cost = cp.Constant(value=0)
    num_measure = len(list_omega)

    list_X = list()
    list_sum_X = list()
    list_sum_rowX = list()
    list_sum_colX = list()

    for i in range(num_measure):
        C = list_C[i]

        X = cp.Variable(C.shape)

        sum_X = cp.sum(X)
        sum_rowX = cp.sum(X, axis=1)
        sum_colX = cp.sum(X, axis=0)

        cost = cp.sum(cp.multiply(X, C))
        entropy_X = cp.sum(cp.entr(X)) + cp.sum(X)
        total_cost += list_omega[i] * (cost - eta * entropy_X)

        list_X.append(X)
        list_sum_X.append(sum_X)
        list_sum_rowX.append(sum_rowX)
        list_sum_colX.append(sum_colX)

    objective = cp.Minimize(total_cost)

    constraints = [list_sum_X[0] == s]
    for i in range(num_measure):
        if i < num_measure - 1:
            constraints += [0 <= list_X[i], list_sum_rowX[i] - list_a[i].reshape(-1, ) <= 0, list_sum_colX[i] - list_sum_colX[i + 1] == 0]
        else:
            constraints += [0 <= list_X[i], list_sum_rowX[i] - list_a[i].reshape(-1, ) <= 0]

    prob = cp.Problem(objective, constraints)
    prob.solve(solver=solver, verbose=verbose)

    if return_status:
        return prob.value, [X.value for X in list_X], prob.status

    return prob.value, [X.value for X in list_X]
