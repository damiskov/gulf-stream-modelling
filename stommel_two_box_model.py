import numpy as np
import matplotlib.pyplot as plt

def set_plot_formatting():
    # Set plot formatting parameters
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.6
    plt.rcParams['axes.labelsize'] = 14

def initialize_variables():
    # Initialize variables
    nn = 0

    # Set model parameters
    R = 2.0        # Absolute value of the ratio of expansion coefficients, x/y
    delta = 1/6    # Conduction rate of salinity with respect to temperature
    lambda_val = 0.2  # Inverse non-dimensional flushing rate
    q = 0.         # Initial flushing rate (0 to 1)
    qdelta = 100.  # Time constant (inertia) for flushing
    yres = 1.      # Steady reservoir y
    resosc = 0.    # Amplitude of reservoir y oscillation
    dtau = 0.01    # Time step of non-dimensional time
    nstep = 1500   # Number of time steps

    yres0 = yres

    ni = 6
    delT = 1/ni
    delS = delT

    return nn, R, delta, lambda_val, q, qdelta, yres, resosc, dtau, nstep, yres0, ni, delT, delS

def simulate_differential_equation(nn, R, delta, lambda_val, q, qdelta, resosc, dtau, nstep, yres0, delT, delS):
    for n1 in np.arange(0, 1 + delT, delT):
        for n2 in np.arange(0, 1 + delS, delS):
            if n1 == 0 or n1 == 1 or n2 == 0 or n2 == 1:
                x, y = simulate_single_case(R, delta, lambda_val, q, qdelta, resosc, dtau, nstep, yres0, n1, n2)
                d = [R * xi - yi for xi, yi in zip(x, y)]
                nn = plot_functions(nn, R, lambda_val,dtau, nstep, x, y, d)

def plot_functions(nn, R, lambda_val, dtau, nstep, x, y, d):
    if nn == 0:
        nn = 1
        plot_initial_case(R, lambda_val, dtau, nstep, x, y, d)
    m2 = len(x)
    if d[m2 - 1] >= 0:
        plt.plot(x, y, 'r--')
        plt.plot(x[m2 - 1], y[m2 - 1], '*r')
    else:
        plt.plot(x, y, 'g')
        plt.plot(x[m2 - 1], y[m2 - 1], '*g')
    return nn

def plot_initial_case(R, lambda_val, dtau, nstep, x, y, d):
    # Make a time series plot for the first case only
    #experiment_1(dtau, nstep, x, y, d) #outcomment this if you want to see the time series plot

    plt.figure(2, figsize=(8,6))

    plt.clf()

    dm = np.zeros((11, 11))

    ym = np.arange(0, 1.1, 0.1)
    xm = np.arange(0, 1.1, 0.1)

    for k1 in range(11):
        for k2 in range(11):
            dm[k1, k2] = (1 / lambda_val) * (R * xm[k2] - ym[k1])

    levels = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]

    c = plt.contour(xm, ym, dm, levels, colors='k')
    plt.clabel(c, inline=True, fontsize=10)
    plt.xlabel('Salinity')
    plt.ylabel('Temperature')
    plt.title("Phase portrait for Stommels Two Box Model")
    plt.ylim(0,1)
    plt.xlim(0,1)
    plt.grid(True)

def experiment_1(dtau, nstep, x, y, d):
    plt.figure(1)
    plt.clf()
    plt.subplot(2, 1, 1)
    plt.plot(np.arange(0, nstep + 1) * dtau, x, '-', np.arange(0, nstep + 1) * dtau, y, '--')
    plt.legend(['Salinity', 'Temperature'])
    plt.ylabel('T, S diff, non-d')
    plt.title('Experiment 1,1')

    plt.subplot(2, 1, 2)
    plt.plot(np.arange(0, nstep + 1) * dtau, d)
    plt.xlabel('time, non-d')
    plt.ylabel('density diff')

def simulate_single_case(R, delta, lambda_val, q, qdelta, resosc, dtau, nstep, yres0, n1, n2):
    x = [n1]  # Initialize x list for this set of initial conditions
    y = [n2]  # Initialize y list for this set of initial conditions

    for m in range(1, nstep + 1):
        tau = m * dtau

        # Evaluate the reservoir temperature (y)
        yres = yres0 + resosc * np.sin(tau * np.pi)

        dr = abs(R * x[m - 1] - y[m - 1])

        # Equilibrium flushing
        qequil = dr / lambda_val

        qequil, yh, xh, qh = update_variables(R, delta, lambda_val, q, qdelta, dtau, x, y, m, yres, qequil)

        y.append(y[m - 1] + dtau * (yres - yh) - dtau * qh * yh)
        x.append(x[m - 1] + dtau * delta * (1 - xh) - dtau * qh * xh)
        q = q + dtau * qdelta * (qequil - qh)
    return x,y

def update_variables(R, delta, lambda_val, q, qdelta, dtau, x, y, m, yres, qequil):
    yh = y[m - 1] + dtau * (yres - y[m - 1]) / 2 - dtau * y[m - 1] * q / 2
    xh = x[m - 1] + dtau * delta * (1 - x[m - 1]) / 2 - dtau * x[m - 1] * q / 2
    qh = q + dtau * qdelta * (qequil - q) / 2

    dr = abs(R * xh - yh)
    qequil = dr / lambda_val
    return qequil,yh,xh,qh

def calculate_and_plot_values(R, delta, lambda_val):
    f = np.zeros(60)
    lhs = np.zeros(60)
    rhs = np.zeros(60)

# Calculate values for f, lhs, and rhs
    for k in range(1, 61):
        f[k - 1] = (k - 30) * 0.1
        lhs[k - 1] = lambda_val * f[k - 1]
        rhs[k - 1] = (R / (1 + abs(f[k - 1]) / delta)) - 1 / (1 + abs(f[k - 1]))

# Create a plot
    plt.figure(3, figsize=(8,6))
    plt.clf()
    plt.plot(f, rhs, f, lhs)
    plt.xlabel('f, flow rate')
    plt.ylabel(r'$\phi(f,R,\delta)$')
    plt.title('Equilibrium states for Stommels Two Box Model')
    plt.grid(True)

def main():
    set_plot_formatting()
    nn, R, delta, lambda_val, q, qdelta, yres, resosc, dtau, nstep, yres0, ni, delT, delS = initialize_variables()
    simulate_differential_equation(nn, R, delta, lambda_val, q, qdelta, resosc, dtau, nstep, yres0, delT, delS)
    calculate_and_plot_values(R, delta, lambda_val)
    plt.show()

main()



