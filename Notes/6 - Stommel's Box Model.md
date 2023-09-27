Previously introduced two-box model with a capillary pipe at the bottom and an overflow mechanism. A virtual salt-flux was used to account for the effects of evaporation, precipitation and runoff from continents.

Various simplifications were made to reduce the dynamics of "overturning circulation" to a single ODE.

In *Stommel's two-box model* a new sequence of simplifications are used to reduce the dynamics to a *planar* dynamical system. (cite Stommel's paper)
- No salt flux at the surface.
- Forcing is entirely due to the exchange of heat and salinity with the atmosphere and neighboring oceans.
## Introduction
---
Variables:
- Temperature anomalies: 
	- $T_1$
	- $T_2$
- Salinity anomalies:
	- $S_1$
	- $S_2$

The temperature and salinity anomalies of the surrounding basins are denoted $T^*$ and $S^*$ respectively.

The governing equations are given by:

Box 1:
- Temperature: $\dfrac{d {T}_1}{d t} = c (-T^{*} - {T}_1) + |q| ({T}_2 - {T}_1)$
- Salinity: $\dfrac{d {S}_1}{d t} = -H + d (-S^{*} - {S}_1) + |q| ({S}_2 - {S}_1)$
Box 2:
- Temperature: $\dfrac{d {T}_2}{d t} = c (-T^{*} - {T}_2) + |q| ({T}_1 - {T}_2)$
- Salinity: $\dfrac{d {S}_2}{d t} = H + d (S^{*} - {S}_2) + |q| ({S}_1 - {S}_2)$

Where $q = k(\alpha(T_2 - T_1) - \beta (S_2 - S_1))$.

Stommel does not account for evaporation or precipitation, so $H = 0$. However, each box does exchange salinity with the surrounding basin, so $d > 0$. This system subsequently reduces to:

Box 1:
- Temperature: $\dfrac{d {T}_1}{d t} = c (-T^{*} - {T}_1) + |q| ({T}_2 - {T}_1)$
- Salinity: $\dfrac{d {S}_1}{d t} = d (-S^{*} - {S}_1) + |q| ({S}_2 - {S}_1)$
Box 2:
- Temperature: $\dfrac{d {T}_2}{d t} = c (-T^{*} - {T}_2) + |q| ({T}_1 - {T}_2)$
- Salinity: $\dfrac{d {S}_2}{d t} = d (S^{*} - {S}_2) + |q| ({S}_1 - {S}_2)$

The mean temperature anomaly $\dfrac{1}{2} (T_1 + T_2)$ and mean salinity anomaly $\dfrac{1}{2}(S_1 + S_2)$ both converge to $0$ as $t \rightarrow \infty$. Introducing differences:
- $\Delta T = T_2 - T_1$
- $\Delta S = S_2 - S_1$

These new variables satisfy the following:

$\dfrac{d \Delta T}{dt} = c (\Delta T^* - \Delta T) - 2 |q| \Delta T$,

$\dfrac{d \Delta S}{dt} = d (\Delta S^* - \Delta S) - 2 |q| \Delta S$

Where $\Delta T^* = 2 T^*$, $\Delta S^* = 2 S^*$  and $q = k(\alpha \Delta T - \beta \Delta S)$. (see chapter 3 again)
## Dynamical System
---
Proceeding with non-dimensionalization using the following variables:

$x = \dfrac{\Delta S}{\Delta S^*}$,   $y = \dfrac{\Delta T}{\Delta T^*}$,

and linear scale:

$t' = c t$

The system of equations therefore becomes:

$\dot{x} = \delta (1-x) - |f|x$,
$\dot{y} = 1 - y - |f|y$,

Where $\delta = \frac{d}{c}$ and $f = - \frac{2 q}{c}$. 

Using the previously defined version of $q$, the relation $f = - \dfrac{2 q}{c}$ can be written as:

$\lambda f(x, y) = R x - y$

where,

$\lambda = \dfrac{c}{2 \alpha k \Delta T^*}$,   $R = \dfrac{\beta \Delta S^*}{\alpha \Delta T^*}$

- $| \lambda |$ is a measure of the strength of the THC
- $R > 1 \implies$ Salinity differences dominate.
- $R < 1 \implies$ Temperature differences dominate.

