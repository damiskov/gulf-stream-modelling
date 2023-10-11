import numpy as np
import matplotlib.pyplot as plt
saving_path = '/Users/karlfindhansen/Desktop/'  
save_plots = False

def set_plot_formatting():
    """
    This function sets the formatting parameters for the plots using plt.rcParams. 
    It adjusts the line width, font size, axes linewidth, and label size for the plots.
    """
    # Set plot formatting parameters
    plt.rcParams['lines.linewidth'] = 1.2
    plt.rcParams['font.size'] = 14
    plt.rcParams['axes.linewidth'] = 1.6
    plt.rcParams['axes.labelsize'] = 14

def initialize_variables():
    """
    This function initializes various model parameters and variables used in the simulation. 
    It defines parameters like R, delta, lambda_val, q, qdelta, yres, and others that govern the behavior of the Stommels Two Box Model. 
    These parameters represent characteristics of the oceanic system being modeled.
    """
    # Initialize variables
    nn = 0

    # Set model parameters
    R = 2.0        # Absolute value of the ratio of expansion coefficients, x/y
    delta = 1/6    # Conduction rate of salinity with respect to temperature
    lambda_val = 2/5  # Inverse non-dimensional flushing rate
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
    """
    This function performs the core simulation of the Stommels Two Box Model. 
    It iterates through different combinations of initial conditions (n1 and n2) and calls simulate_single_case() for each combination. 
    It also calls plot_functions() to create plots based on the simulation results.
    """
    for n1 in np.arange(0, 1 + delT, delT):
        for n2 in np.arange(0, 1 + delS, delS):
            if n1 == 0 or n1 == 1 or n2 == 0 or n2 == 1:
                x, y = simulate_single_case(R, delta, lambda_val, q, qdelta, resosc, dtau, nstep, yres0, n1, n2)
                d = [R * xi - yi for xi, yi in zip(x, y)]
                nn = plot_functions(nn, R, lambda_val,dtau, nstep, x, y, d)

def plot_functions(nn, R, lambda_val, dtau, nstep, x, y, d):
    """
    This function creates plots to visualize the results of the simulation. 
    It plots the trajectories of the model's two variables (x and y) on a phase plane, with different markers and colors to represent different cases. 
    It also marks specific points on the plot (labeled 'a', 'b', 'c').
    """
    if nn == 0:
        nn = 1
        # Make a time series plot for the first case only
        #experiment_1(dtau, nstep, x, y, d) #outcomment this if you want to see the time series plot
        plot_initial_case(R, lambda_val)
    m2 = len(x)
    if d[m2 - 1] >= 0:
        plt.plot(x, y, 'r--')
        plt.plot(x[m2 - 1], y[m2 - 1], '*r')
        plt.text(x[m2 - 1], y[m2 - 1], "c")
    else:
        plt.plot(x, y, 'g')
        plt.plot(x[m2 - 1], y[m2 - 1], '*g')
        plt.text(x[m2 - 1], y[m2 - 1], "a")
    if lambda_val == 1/5:
        plt.plot(0.347, 0.777, '*k')
        plt.text(0.347, 0.777, "b")
    if save_plots:
        plt.savefig(saving_path+'phase_portrait.png', dpi=400)
    return nn

def plot_initial_case(R, lambda_val):
    """
    This function generates the initial phase portrait for the Stommels Two Box Model. 
    It creates a contour plot to visualize the equilibrium states of the system.
    """
    plt.figure(2, figsize=(8,6))

    plt.clf()

    dm = np.zeros((11, 11))

    ym = np.arange(0, 1.1, 0.1)
    xm = np.arange(0, 1.1, 0.1)

    for k1 in range(11):
        for k2 in range(11):
            dm[k1, k2] = (1 / lambda_val) * (R * xm[k2] - ym[k1])

    levels = [-10, -8, -6, -4, -2, 0, 2, 4, 6, 8, 10, 12, 14, 16, 18, 20]
    contour_colors = ['k' if level != 0 else 'k' for level in levels]

    c = plt.contour(xm, ym, dm, levels, colors=contour_colors)
    plt.clabel(c, inline=True, fontsize=10)
    plt.xlabel('Salinity')
    plt.ylabel('Temperature')
    plt.title("Phase portrait for Stommels Two Box Model")
    plt.ylim(0,1)
    plt.xlim(0,1)

    plt.grid(True)

def experiment_1(dtau, nstep, x, y, d):
    """
    This function generates time series plots for the first case of the simulation. 
    It plots the evolution of the variables over time.
    """
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
    """
    This function simulates a single case of the Stommels Two Box Model given initial conditions (n1 and n2). 
    It iteratively calculates the values of the model variables (x and y) over a specified number of time steps.
    """
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
    """
    This function updates the variables in each time step of the simulation based on the model equations.
    """
    yh = y[m - 1] + dtau * (yres - y[m - 1]) / 2 - dtau * y[m - 1] * q / 2
    xh = x[m - 1] + dtau * delta * (1 - x[m - 1]) / 2 - dtau * x[m - 1] * q / 2
    qh = q + dtau * qdelta * (qequil - q) / 2

    dr = abs(R * xh - yh)
    qequil = dr / lambda_val
    return qequil,yh,xh,qh

def calculate_and_plot_values(R, delta, lambda_val):
    """
    This function calculates and plots equilibrium states for the model by evaluating the functions phi(f, R, delta) and lambda*f. 
    It also marks the equilibrium flow rates on the plot.
    """
    t = 2000
    f = np.zeros(t)
    lhs = np.zeros(t)
    rhs = np.zeros(t)

    lhs1 = np.zeros(t)
    lhs2 = np.zeros(t)

# Calculate values for f, lhs, and rhs
    for k in range(1, t+1):
        f[k - 1] = (k - 30) * 0.1
        lhs[k - 1] = lambda_val * f[k - 1]
        rhs[k - 1] = (R / (1 + abs(f[k - 1]) / delta)) - 1 / (1 + abs(f[k - 1]))
        ## add new lines here for new lambda values
        lhs1[k-1] = 1 * f[k - 1]
        lhs2[k-1] = 2/5 * f[k - 1]

    ins_plot1 = (-1.061,-0.22)
    ins_plot2 = (-0.3, -0.07)
    ins_plot3 = (0.23, 0.032)

    intersection_points = [ins_plot1, ins_plot2, ins_plot3]
# Create a plot
    plt.figure(3, figsize=(8,6))
    plt.clf()
    plt.plot(f, rhs, linewidth = 1.5, color = 'k') #label=r'$f \rightarrow \phi(f,R,\delta)$ with $R=2$ and $\delta = \frac{1}{6}$'
    plt.plot(f, lhs,linewidth = 1.5 ,color = 'blue') #label=r'$f \rightarrow \lambda f$ with $\lambda = \frac{1}{5}$'
    plt.plot(f, lhs1,linewidth = 1.5, color = 'red') #label=r'$f \rightarrow f$ with $\lambda = 1$'
    plt.plot(f, lhs2,linewidth = 1.5, color = 'green') #label=r'$f \rightarrow f$ with $\lambda = 2/5$'

    for intersection_point in intersection_points:
        plt.plot(intersection_point[0], intersection_point[1], marker='o', color='blue')
    
    arrow_props = dict(facecolor='black', arrowstyle='->', linewidth=1.2)
    for i, intersection_point in enumerate(intersection_points):
        #plt.annotate(f'Intersection {i+1}', intersection_point, textcoords='offset points', xytext=(0,-20), ha='center', fontsize=12)
        if i == 0:
            plt.annotate('', xy=(intersection_point[0], intersection_point[1]), xytext=(intersection_point[0]-0.11, intersection_point[1]+0.11), arrowprops=arrow_props)
            plt.annotate(f'$f_{i+1}$', (intersection_point[0]-0.14, intersection_point[1]+0.12), fontsize=12)
        if i == 1:
            plt.annotate('', xy=(intersection_point[0], intersection_point[1]), xytext=(intersection_point[0]-0.11, intersection_point[1]+0.11), arrowprops=arrow_props)
            plt.annotate(f'$f_{i+1}$', (intersection_point[0]-0.14, intersection_point[1]+0.12), fontsize=12)
        if i == 2:
            plt.annotate('', xy=(intersection_point[0], intersection_point[1]), xytext=(intersection_point[0]-0.11, intersection_point[1]-0.11), arrowprops=arrow_props)
            plt.annotate(f'$f_{i+1}$', (intersection_point[0]-0.16, intersection_point[1]-0.16), fontsize=12)


    plt.plot(intersection_point[0], intersection_point[1], marker='o', color='blue', label = 'Equilibrium flow rates')

    plt.xlabel(r'flow rate $f$')
    plt.ylabel(r'$\phi(f,R,\delta)$')
    plt.title('Equilibrium states for Stommels Two Box Model')
    plt.legend()
    plt.ylim(-0.5,1.2)
    plt.xlim(-2,2)
    if save_plots:
        plt.savefig(saving_path+'equilibrium_states.png', dpi=400)

def main():
    set_plot_formatting()
    nn, R, delta, lambda_val, q, qdelta, yres, resosc, dtau, nstep, yres0, ni, delT, delS = initialize_variables()
    simulate_differential_equation(nn, R, delta, lambda_val, q, qdelta, resosc, dtau, nstep, yres0, delT, delS)
    calculate_and_plot_values(R, delta, lambda_val)
    plt.show()

main()



