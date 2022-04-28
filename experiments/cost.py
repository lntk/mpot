import numpy as np
from scipy.special import factorial

D0 = 100  # D_{0}
m = 20
list_delta2 = [-factorial(m - 1 - i) for i in range(1, m - 1)]
delta1 = (sum([-(m - 1 - i) * list_delta2[i - 1] for i in range(1, m - 1)]) + D0) / (m - 1)

list_D = [D0]
list_D += [sum([(i - j) * list_delta2[j - 1] for j in range(1, i)]) + i * delta1 + D0 for i in range(1, m - 1)]
list_D += [0, 1]  # D_{m - 1}, D_{m}
print(list_D)

""" (*.*) Plotting (-.-) """
import matplotlib.pyplot as plt

fig, axs = plt.subplots(1, 1, figsize=(5, 5), sharex='all')
axs.plot(list(range(m + 1)), list_D, 'r', label='D')

plt.legend(prop={'size': 15}, loc='best')
plt.tight_layout()

# plt.savefig(f'./(gd_vs_agd)_(L={L})_(mu={mu}).pdf', bbox_inches='tight')

plt.show()
plt.close()
