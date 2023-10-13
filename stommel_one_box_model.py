import numpy as np
import matplotlib.pyplot as plt

# Stommel's one-box model

tau = np.arange(0, 100, 0.1)

x = lambda t: 1 - np.exp(-t/6)
y = lambda t: 1 - np.exp(-t)

y_sigma = lambda sigma, t: 2*x(t)-sigma

markers = np.array([1,2,3,4,5,100])

# Temperature vs Salinity
plt.plot(x(tau), y(tau))

plt.xlim(0,1)
plt.ylim(0,1)

# Density anomalies
plt.plot(x(tau), y_sigma(-0.5, tau))
plt.plot(x(tau), y_sigma(0, tau))

plt.plot(x(tau), y_sigma(0.5, tau))
plt.plot(x(tau), y_sigma(1, tau))


# Annotations

plt.annotate('\u03C3=-1/2', xy=(0.067, 0.84))
plt.annotate('\u03C3=0', xy=(0.33, 0.615))
plt.annotate('\u03C3=1/2', xy=(0.60, 0.615))
plt.annotate('\u03C3=1', xy=(0.85, 0.615))



# Time markers
plt.scatter(x(markers), y(markers))


plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.show()

