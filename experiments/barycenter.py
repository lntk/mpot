""" Author: Khang Le """

###################### HEADER #######################
import sys
from os.path import dirname, abspath

main_dir = dirname(dirname(abspath(__file__)))
print(f"Main Dir: {main_dir}")
sys.path.append(main_dir)
#####################################################

""" --- Import --- """
import itertools
import numpy as np

import ot

from optimization.cp import solve_entropic_mpot, solve_entropic_ot_barycenter, solve_entropic_pot_barycenter

from utils.metric import euclidean_distance_numpy
from utils.general import save_object, read_object


def flatten(X, m, n):
    return [X[indices] for indices in itertools.product(*[tuple(range(n)) for _ in range(m)])]


def process(y, p):
    y = np.array(y)
    p = np.array(p)

    y_unique, indices = np.unique(y, return_inverse=True)
    num_unique = y_unique.shape[0]

    p_new = np.zeros_like(y_unique)

    for i in range(num_unique):
        p_new[i] += np.sum(p[np.where(indices == i)])

    return y_unique.tolist(), p_new.tolist()


def binning(y, p, n):
    p_binned = list()
    j = 0
    for i in range(n):
        bin_value = 0
        while i - 0.5 <= y[j] <= i + 0.5:
            bin_value += p[j]
            j += 1
            if j >= len(y):
                break
        p_binned.append(bin_value)
    return p_binned


""" --- Setup --- """
n = 100
m = 3
x = np.arange(n, dtype=np.float64)

list_omega = [0.1, 0.2, 0.7]
weights = np.array(list_omega)

outlier_ratio = 0.1

""" --- Histograms --- """
a1_true = ot.datasets.make_1D_gauss(n, m=50, s=5)
a1_outlier = ot.datasets.make_1D_gauss(n, m=5, s=2)
a1 = (1 - outlier_ratio) * a1_true + outlier_ratio * a1_outlier

a2_true = ot.datasets.make_1D_gauss(n, m=45, s=5)
a2_outlier = ot.datasets.make_1D_gauss(n, m=90, s=2)
a2 = (1 - outlier_ratio) * a2_true + outlier_ratio * a2_outlier

a3_true = ot.datasets.make_1D_gauss(n, m=55, s=3)
a3_outlier = ot.datasets.make_1D_gauss(n, m=10, s=3)
a3 = (1 - outlier_ratio) * a3_true + outlier_ratio * a3_outlier

run_exp = False
if run_exp:
    """ OT Barycenter """
    C = euclidean_distance_numpy(x.reshape(-1, 1), x.reshape(-1, 1), squared=True)
    list_C = [C for _ in range(m)]
    list_a = [a1.reshape(-1, 1), a2.reshape(-1, 1), a3.reshape(-1, 1)]

    _, list_X, _ = solve_entropic_ot_barycenter(list_omega=list_omega, list_C=list_C, list_a=list_a, eta=1., verbose=False, return_status=True, solver="MOSEK")
    barycenter_ot = np.sum(list_X[0], axis=0)

    """ POT Barycenter """
    C = euclidean_distance_numpy(x.reshape(-1, 1), x.reshape(-1, 1), squared=True)
    list_C = [C for _ in range(m)]
    list_a = [a1.reshape(-1, 1), a2.reshape(-1, 1), a3.reshape(-1, 1)]

    s = 1 - 2 * outlier_ratio

    _, list_X, _ = solve_entropic_pot_barycenter(list_omega=list_omega, list_C=list_C, list_a=list_a, s=s, eta=1., verbose=False, return_status=True, solver="MOSEK")
    barycenter_pot = np.sum(list_X[0], axis=0)

    """ MPOT Barycenter """
    C = np.empty(shape=((n,) * m))
    y = list()

    for indices in itertools.product(*[tuple(range(n)) for _ in range(m)]):  # loop over all possible j=(j1, j2, ..., jk) \in [n]^m
        xj = [x[i] for i in indices]  # xj = [x_j1, x_j2, ..., x_jk]
        yj = np.sum(weights * xj)  # yj = <weights, xj>

        C[indices] = 1 / 2 * np.sum(weights * (xj - yj) ** 2)  # C[j1, j2, ..., jk] = 1/2 * < weights, ||x - A||_2^2 >
        y.append(yj)

    sorted_indices = np.argsort(y)
    y = np.array(y)[sorted_indices].tolist()

    s = 1 - 2 * outlier_ratio

    _, P_empot = solve_entropic_mpot(C, list_a, s, eta=1, verbose=True)

    p_empot = flatten(P_empot, m, n)
    p_empot = np.array(p_empot)[sorted_indices].tolist()

    y_empot_new, p_empot_new = process(y, p_empot)
    p_empot_new_binned = binning(y_empot_new, p_empot_new, n)

    save_object({
        'barycenter_ot': barycenter_ot,
        'barycenter_pot': barycenter_pot,
        'p_empot_new_binned': p_empot_new_binned,
    }, f'{main_dir}/data/barycenter.data')
else:
    data = read_object(f'{main_dir}/data/barycenter.data')
    barycenter_ot = data['barycenter_ot']
    barycenter_pot = data['barycenter_pot']
    p_empot_new_binned = data['p_empot_new_binned']

""" --- Plotting --- """
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

fig, axs = plt.subplots(1, 2, figsize=(10, 5), sharex='all')

axs[0].plot(x, a1)
axs[0].plot(x, a2)
axs[0].plot(x, a3)
axs[0].set_title('Distributions')

axs[1].plot(x, barycenter_ot, 'r', label='BC-OT')
axs[1].plot(x, barycenter_pot, 'g', label='BC-POT')
axs[1].plot(x, p_empot_new_binned, 'b', label='BC-MPOT')
axs[1].set_title('Barycenters')
# plt.plot(y_new, p_new, 'r', label='MOT')

plt.legend(prop={'size': 15}, loc='best')
plt.tight_layout()

plt.savefig(f'{main_dir}/logs/barycenter_mpot.pdf', bbox_inches='tight')

plt.show()
