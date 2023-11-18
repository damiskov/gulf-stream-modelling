import matplotlib.pyplot as plt

# Data from https://www.usgs.gov/special-topics/water-science-school/science/water-density#overview

temperature = [0, 4, 4.4, 10, 15.6, 21, 26.7, 32.2, 37.8, 48.9, 60, 71.1, 82.2, 93.3, 100]

density = [999.87, 1000, 999.99, 999.75, 999.07, 998.02, 996.69, 995.10, 993.18, 988.70, 983.38, 977.29, 970.56, 963.33, 958.65]

plt.plot(temperature, density, marker='x', color='b')
plt.xlabel('Temperature (\u00B0 C)')
plt.ylim(950, 1005)
plt.grid()
plt.rcParams["figure.dpi"] = 600
plt.show()
