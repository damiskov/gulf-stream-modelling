from matplotlib import pyplot
import numpy

import phaseportrait

def pendulum(θ, dθ):
    return dθ, - numpy.sin(θ)

SimplePendulum = phaseportrait.PhasePortrait2D(pendulum, [-9, 9], Title='Simple pendulum', xlabel=r"$\Theta$", ylabel=r"$\dot{\Theta}$")
SimplePendulum.plot()
