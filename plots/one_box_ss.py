import matplotlib.pyplot as plt
import numpy as np

# Stommel's one-box model
x = np.arange(0, 1, 0.01)
f = lambda x: (6*x)/(1+5*x)
y = f(x)
x_vals = np.array([100, 4, 3, 2, 1, 1/2, 1/4, 1/10, 0])
f_prime = 6*x_vals/(1+5*x_vals)
y_sigma = lambda sigma, x: 2*x-sigma

# Temperature vs Salinity
plt.plot(x, y)
plt.xlim(0,1)
plt.ylim(0,1)
plt.scatter(x_vals, f_prime)

# Density anomalies
plt.plot(x, y_sigma(-0.5, x))
plt.plot(x, y_sigma(0, x))
plt.plot(x, y_sigma(0.5, x))
plt.plot(x, y_sigma(1, x))
plt.plot(x, y_sigma(1.5, x))

# f prime annotation
plt.annotate('f\' \u2192 0', xy=(0.87, 0.91))


plt.show()