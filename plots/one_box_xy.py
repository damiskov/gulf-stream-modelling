import numpy as np
import matplotlib.pyplot as plt

# Stommel's one-box model

tau = np.arange(0, 100, 0.1)

x = lambda t: 1 - np.exp(-t/6)
y = lambda t: 1 - np.exp(-t)

y_sigma = lambda sigma, t: 2*x(t)-sigma

markers = np.array([0,1,2,3,4,5,100])

# Temperature vs Salinity
plt.plot(x(tau), y(tau))

plt.xlim(0,1)
plt.ylim(0,1)

# Density anomalies
plt.plot(x(tau), y_sigma(-0.5, tau))
plt.plot(x(tau), y_sigma(0, tau))
plt.plot(x(tau), y_sigma(0.5, tau))
plt.plot(x(tau), y_sigma(1, tau))
plt.plot(x(tau), y_sigma(1.5, tau))


# tau annotations

plt.annotate('\u03C4  \u2192 \u221E', xy=(0.88, 0.95))
plt.annotate('\u03C4=0', xy=(0.04, 0.025))
plt.annotate('1', xy=(0.168, 0.626))
plt.annotate('2', xy=(0.303, 0.842))
plt.annotate('3', xy=(0.401, 0.915))
plt.annotate('4', xy=(0.498, 0.952))
plt.annotate('5', xy=(0.57, 0.959))

# Denisity anomaly annotations
plt.annotate('\u03C3=-1/2', xy=(0.1, 0.84), rotation=60)
plt.annotate('\u03C3=0', xy=(0.33, 0.615), rotation=60)
plt.annotate('\u03C3=1/2', xy=(0.60, 0.615), rotation=60)
plt.annotate('\u03C3=1', xy=(0.85, 0.615), rotation=60)
plt.annotate('\u03C3=3/2', xy=(0.9, 0.222), rotation=60)


# Time markers
plt.scatter(x(markers), y(markers))


plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.show()

