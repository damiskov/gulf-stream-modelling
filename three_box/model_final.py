import matplotlib.pyplot as plt
import numpy as np
from phaseportrait import PhasePortrait3D

psi = 0.1

def three_box(x_N, x_L, x_E, psi=psi):
    psi=0.1
    x_N_n = -psi/2 - x_N - 1/2 - 2*abs(-127 - 150*x_N + 150*x_E)*(x_E - x_N)
    x_L_n = -psi/2 - x_L - 1/2 + 2*abs(-269 - 150*x_L + 150*x_E)*(x_E - x_L)
    x_E_n = psi + 1 - x_E - 2*abs(-269 - 150*x_L + 150*x_E)*(x_E - x_L)
    return x_N_n, x_L_n, x_E_n

plt.rcParams['figure.figsize'] = [20, 10]
plt.rcParams['figure.dpi'] = 600


three_box_portrait = PhasePortrait3D(three_box, [-5, 5], MeshDim=8,Title=r'', xlabel=r"$x_N$", ylabel=r"$x_L$", zlabel=r"$x_E$")
three_box_portrait.plot()
plt.show()
