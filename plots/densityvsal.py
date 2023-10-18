import matplotlib.pyplot as plt
import numpy as np

# Density and salinity data (https://www.jodc.go.jp/jodcweb/info/ioc_doc/UNESCO_tech/096451mb.pdf)

sal = np.arange(42, 46, 0.1)

density_s = [31.112, 31.189, 31.266, 31.343, 31.420,
           31.497, 31.574, 31.651, 31.728, 31.805,
           31.882, 31.959, 32.036, 32.113, 32.190,
           32.267, 32.344, 32.421, 32.498, 32.575,
           32.652, 32.729, 32.806, 32.883, 32.960,
           33.037, 33.114, 33.192, 33.269, 33.346,
           33.423, 33.500, 33.577, 33.654, 33.731,
           33.808, 33.886, 33.963, 34.040, 34.117]



# Have 2 subfigures, one for salinity and one for temperature, horizontally stacked



# plt.plot(sal, density, marker='x', color='r')
# plt.xlabel('Salinity (psu)')
# plt.ylabel('\u0394 \u03C1 (g/m\u00B3)')
# plt.grid()
# plt.rcParams["figure.dpi"] = 600


temperature = [0, 4, 4.4, 10, 15.6, 21, 26.7, 32.2, 37.8, 48.9, 60, 71.1, 82.2, 93.3, 100]

density_t = [999.87, 1000, 999.99, 999.75, 999.07, 998.02, 996.69, 995.10, 993.18, 988.70, 983.38, 977.29, 970.56, 963.33, 958.65]

# plt.plot(temperature, density, marker='x', color='b')
# plt.xlabel('Temperature (\u00B0 C)')
# plt.ylim(950, 1005)
# plt.grid()
# plt.rcParams["figure.dpi"] = 600
# plt.show()

plt.subplot(1, 2, 1)
plt.plot(sal, density_s, marker='x', color='b')
plt.xlabel('Salinity (psu)')
plt.ylabel('\u0394 \u03C1 (g/m\u00B3)')
plt.grid()
plt.rcParams["figure.dpi"] = 600

plt.subplot(1, 2, 2)
plt.plot(temperature, density_t, marker='x', color='r')
plt.xlabel('Temperature (\u00B0 C)')
plt.ylabel('\u03C1 (kg/m\u00B3)')
plt.ylim(950, 1005)
plt.grid()
plt.show()

