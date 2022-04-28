""" Author: Khang Le """

###################### HEADER #######################
import sys
from os.path import dirname, abspath

main_dir = dirname(dirname(abspath(__file__)))
print(f"Main Dir: {main_dir}")
sys.path.append(main_dir)
#####################################################

""" --- Import --- """
from optimization.cp import solve_mpot
from optimization.sinkhorn import sinkhorn_mpot

from utils.general import save_object, read_object
from utils.data import init_measure, init_cost

""" --- Setup --- """
n = 10
m = 3
s = 0.8

a1 = init_measure(n, alpha=1)
a2 = init_measure(n, alpha=1)
a3 = init_measure(n, alpha=1)

list_a = [a1, a2, a3]

C = init_cost(size=(n, n, n), symmetric=False)

list_logs = list()
list_eta = [0.01, 0.1, 1.]

run_exp = False
if run_exp:
    for eta in list_eta:
        _, _, logs = sinkhorn_mpot(C, list_a, s, eta, epsilon=0.01, max_iter=1000, logging=True, verbose=False)
        list_logs.append(logs)

    value_cp, X_mpot_cp = solve_mpot(C, list_a, s)

    save_object({
        'value_cp': value_cp,
        'list_logs': list_logs,
    }, f'{main_dir}/data/convergence.data')
else:
    data = read_object(f'{main_dir}/data/convergence.data')
    value_cp = data['value_cp']
    list_logs = data['list_logs']

""" --- Plotting --- """
import matplotlib.pyplot as plt
import matplotlib

matplotlib.rcParams['pdf.fonttype'] = 42
matplotlib.rcParams['ps.fonttype'] = 42

matplotlib.rcParams.update({'font.size': 22})
fig, axs = plt.subplots(1, 1, figsize=(10, 5), sharex='all')

for i in range(len(list_logs)):
    logs = list_logs[i]
    axs.plot(logs['iterations'], logs['values'], label=r'ApproxMPOT ~ $\eta=$' + f'{list_eta[i]}')

axs.plot([0, 1000], [value_cp, value_cp], label='Optimal')
axs.set_xlabel('Iterations')
axs.set_ylabel('Objective Values')

plt.legend(prop={'size': 25}, loc='best')
plt.tight_layout()

plt.savefig(f'{main_dir}/logs/convergence.pdf', bbox_inches='tight')

plt.show()
