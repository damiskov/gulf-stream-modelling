import matplotlib.pyplot as plt
import numpy as np

# Stommel's one-box, steady-state model
x = np.arange(0, 1, 0.01)
x_f_delta = lambda delta, f: 1/(1+f/delta)
y_f = lambda f: 1/(1+f)
f_y = lambda x, delta: x/(delta + x*(1-delta))
f_vals = np.array([100, 4, 1, 1/2, 1/4, 1/20, 0])
y_sigma = lambda sigma, x: 2*x-sigma

# Using delta = 1/6

delta = 1/6

y = f_y(x, delta)

x_vals = x_f_delta(delta, f_vals)
y_vals = y_f(f_vals)



# Temperature vs Salinity
plt.plot(x, y, c='orange')
plt.xlim(0,1)
plt.ylim(0,1)

# Scatter plot x, y at certain f-values, labeled

plt.scatter(x_vals, y_vals, marker='o', c='orange')
labels = ['\u221E', '4', '1', '1/2', '1/4', '1/20', '0']
n = 0
for i, j in zip(x_vals, y_vals):
    plt.annotate(labels[n], xy=(i, j), xytext=(i-0.01, j+0.02))
    n += 1





# Density anomalies
plt.plot(x, y_sigma(-0.5,x), c='k', ls='--')
plt.plot(x, y_sigma(0, x), c='k', ls='--')
plt.plot(x, y_sigma(0.5, x), c='k', ls='--')
plt.plot(x, y_sigma(1, x), c='k', ls='--')
plt.plot(x, y_sigma(1.5, x), c='k', ls='--')

plt.xlabel('Salinity')
plt.ylabel('Temperature')

# save figure with 600 dpi

plt.savefig('one_box_ss.png', dpi=600)

plt.show()