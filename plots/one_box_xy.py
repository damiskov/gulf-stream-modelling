import numpy as np
import matplotlib.pyplot as plt

# Stommel's one-box model

tau = np.arange(0, 100, 0.1)

x = lambda t: 1 - np.exp(-t/6)
y = lambda t: 1 - np.exp(-t)

y_sigma = lambda sigma, t: 2*x(t)-sigma

markers = np.array([0,1,2,3,4,5,100])

# Temperature vs Salinity, green graph
plt.plot(x(tau), y(tau), c='orange')

plt.xlim(0,1)
plt.ylim(0,1)

# Density anomalies
# make lines dashed
plt.plot(x(tau), y_sigma(-0.5, tau), c='k', ls='--')
plt.plot(x(tau), y_sigma(0, tau), c='k', ls='--')
plt.plot(x(tau), y_sigma(0.5, tau), c='k', ls='--')
plt.plot(x(tau), y_sigma(1, tau), c='k', ls='--')
plt.plot(x(tau), y_sigma(1.5, tau), c='k', ls='--')


# tau annotations

plt.annotate('\u03C4  \u2192 \u221E', xy=(0.88, 0.95))
plt.annotate('\u03C4=0', xy=(0.04, 0.025))
plt.annotate('1', xy=(0.168, 0.624))
plt.annotate('2', xy=(0.303, 0.840))
plt.annotate('3', xy=(0.401, 0.912))
plt.annotate('4', xy=(0.498, 0.948))
plt.annotate('5', xy=(0.57, 0.952))

# Denisity anomaly annotations

offset = 0.03
plt.annotate('\u03C3=-1/2', xy=(1/8-offset, 3/4), rotation=60)
plt.annotate('\u03C3=0', xy=(1/4-offset, 1/2), rotation=60)
plt.annotate('\u03C3=1/2', xy=(1/2-offset, 1/2), rotation=60)
plt.annotate('\u03C3=1', xy=(3/4-offset, 1/2), rotation=60)
plt.annotate('\u03C3=3/2', xy=(7/8-offset, 1/4), rotation=60)


# Time markers, orange color
plt.scatter(x(markers), y(markers), marker='o', c='orange')





plt.xlabel('Salinity')
plt.ylabel('Temperature')
plt.savefig('one_box_xy.png', dpi=600)
plt.show()

