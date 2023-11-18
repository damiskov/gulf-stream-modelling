------
## 3.1 - Circulation

- Global ocean circulation patterns are driven by density contrasts
	- Density is a function of salinity and temperature, so the pattern is known as the *thermohaline circulation* (THC)
- Current *conveyor belt* model first suggested by **Broecker**. (Reference developments made)
- Circulatory system provides a stabilizing effect on global climate due to timeframe of water movement - However, if disturbed it can bring about significant changes in the earth's climate (*reference to paleoclimatic modelling paper*).
- A deep understanding of the current configuration and it's stability is therefore critical for global climatic modelling.

![[conveyor_belt.jpg|500]]

-----
### 3.2 - Temperature

- (See cross-section of Atlantic ocean)
	- Radiative heat from the sun only significant for the top few meters - the *mixing layer*, where temperature is essentially constant.
	- Intermediate *thermocline* region separates mixing layer from deep *abyssal zone* (abyssal zone comprises $\approx$ 98% of total oceanic volume)
	- Essentially: Thin upper layer of "warm" constant temperature. Going through thermocline region, the temperature decreases linearly. Final abyssal zone has low constant temperature (just above freezing).

Temperature in Thermocline region changes due to upwelling/advection of cold water and diffusion due to small eddies.

abyssal zone extends to surface at ice caps. Ice formed at extreme latitudes is nearly fresh, leaving higher salt concentration in water and reducing it's freezing point. Cold, salty brine is dense and sinks to the bottom only returning to the surface at lower latitudes centuries later.

![[atlantic_circulation.jpg|500]]
#### Simple one-dimensional advection-diffusion equation

- $\frac{\partial T}{\partial t} = -w \frac{\partial T}{\partial z} + x \frac{\partial^2 T}{\partial z^2}$

At steady state $\frac{\partial T}{\partial t} = 0$. 

$T(z) = T_0 + T_1 \cdot e^{- \frac{z}{z*}}$

Where:
- $w$ is advection rate (~ $10^{-4}$)
- $x$ is the diffusion coefficient (~ $10^{-2}$)
- $z^* = \frac{x}{w}$ is the characteristic depth for the temperature profile.

-------
### 3.3 - Salinity 

- Dissolved salts have a large effect on water density.
- Sea water $\approx$ 3.5% salt.
- Ratios of various ions is relatively constant indicating that the oceans are well mixed.
- Density relies on temperature and salinity, but there is no equation. (Must apply empirical formulae from experimental data)

![[water_density.jpg|500]]

----
### 3.4 - Box Models

Modelling THC is challenging:
- Lack of universal equation of state linking water density to salinity and temperature.
- Complex domain, bounded by various continents.

Massively simplified system-level approach:
- Oceans are just reservoirs of salty water.
- Circulation driven by density differences.

#### North Atlantic two-box model

![[two_box_NA.jpg|500]]

- Temperature and Salinity are the uniform in each box, but not necessarily the same. Leading to density differences.
- Density difference drives flow in lower capillary pipe. Compensatory flow in upper capillary pipe to ensure constant volume.
- Wind and Coriolis effects are ignored.
- Virtual Salt flux represented by $H$
	- Water evaporating dominated box 2, leading to a virtual salt flux into box 2.
	- Precipitation and runoff dominates at colder, higher latitudes leading to a virtual salt flux out of box 1.
- Constant temperature salinity in the basins surrounding each box are represented by: $T^*$  and  $S^*$

#### Mathematical Modelling

Flow $q$ through the capillary pipe is given by pressure differences, which is proportional to the density differences:

$q = k \cdot \dfrac{\rho_1 - \rho_2}{\rho_0}$

-  $\rho_0$: is a reference density.
- $k$: Hydraulic constant (~$1.5 \cdot 10^{-6} \ \text{s}^{-1}$)

In order to connect density and salinity, and equation of state is needed: $\rho : (T, S) \mapsto \rho(T, S)$

Examining **figure 3.6** we assume that $\rho$ varies linearly with $T$ and $S$ near their average values.

- Thermal expansion causes a decrease in density.
- An increase in salt concentration results in an increase in density.

Construct the approximate equation of state:

$\rho_1 = \rho_0 (1- \alpha (T_1-T_0) + \beta (S_1-S_0))$

and 

$\rho_2 = \rho_0 (1- \alpha (T_2-T_0) + \beta (S_2-S_0))$

Substituting these two into the equation for $q$ results in the following relationship between flow, temperature and salinity:

$q = k (\alpha(T_2 - T_1) - \beta (S_2 - S_1)) = k (\alpha \Delta T - \beta \Delta S)$

The governing equations for temperature and salinity are essentially the conservation laws for heat and salinity:

- Box 1 (Lower latitudes):
	- Temperature: $\dfrac{d T_1}{d t} = c (T^{*}_{1} - T_1) + |q| (T_2 - T_1)$
	- Salinity:  $\dfrac{d S_1}{d t} = -H + d (S^{*}_{1} - S_1) + |q| (S_2 - S_1)$
- Box 2 (Extreme latitudes):
	- Temperature: $\dfrac{d T_2}{d t} = c (T^{*}_{2} - T_2) + |q| (T_1 - T_2)$
	- Salinity:  $\dfrac{d S_2}{d t} = H + d (S^{*}_{2} - S_2) + |q| (S_1 - S_2)$

Now consider averages of temperature and salinity: $T_0 = \dfrac{1}{2} (T_1 + T_2)$ and $S_0 = \dfrac{1}{2} (S_1 + S_2)$. It follows from the above equations that $T_0$ and $S_0$ satisfy the equations:

- $\dfrac{d T_0}{d t} = c (T^{*}_{0} - T_0)$
- $\dfrac{d S_0}{d t} = d (S^{*}_{0} - S_0)$

These equations show that both the average salinity and temperature tends to that of the surrounding basins as $t \rightarrow \infty$. This suggests that we take the average temperature and salinity values of the surrounding basins as reference values and introduce the temperature and salinity *anomalies*.

Temperature:
- $\bar{T}_1 = T_1 - T^{*}_0$
- $\bar{T}_2 = T_2 - T^{*}_0$
Salinity:
- $\bar{S}_1 = S_1 - S^{*}_0$
- $\bar{S}_2 = S_2 - S^{*}_0$

This further simplifies after the introduction of the quantities $T^{*} = \dfrac{1}{2}(T^{*}_{2} - T^{*}_{1})$ and $S^{*} = \dfrac{1}{2}(S^{*}_{2} - S^{*}_{1})$.

Re-writing the original system of equations in terms of anomalies and average surrounding basin values:

Box 1 (Lower latitudes):
- Temperature: $\dfrac{d \bar{T}_1}{d t} = c (-T^{*} - \bar{T}_1) + |q| (\bar{T}_2 - \bar{T}_1)$
- Salinity: $\dfrac{d \bar{S}_1}{d t} = -H + d (-S^{*} - \bar{S}_1) + |q| (\bar{S}_2 - \bar{S}_1)$
Box 2 (Extreme latitudes):
- Temperature: $\dfrac{d \bar{T}_2}{d t} = c (-T^{*} - \bar{T}_2) + |q| (\bar{T}_1 - \bar{T}_2)$
- Salinity: $\dfrac{d \bar{S}_2}{d t} = H + d (S^{*} - \bar{S}_2) + |q| (\bar{S}_1 - \bar{S}_2)$

The advantage of this new system (in terms of anomalies and averages) is that we are only dealing with single constants $S^{*}$ and $T^*$.

#### One-Dimensional Model

- Ignore temperature equations.
- Temperature in each box equilibrates almost instantaneously with the surrounding basin.
- Difference between temperatures in the two boxes is small.
- Assume that salinity exchange between the boxes and basins is negligible ($d = 0$).

The system now reduces to:

-  $\dfrac{d \bar{S}_1}{d t} = -H +  |q| (\bar{S}_2 - \bar{S}_1)$
- $\dfrac{d \bar{S}_2}{d t} = H + |q| (\bar{S}_1 - \bar{S}_2)$

Where $q = k ( 2 \alpha T^{*} - \beta (\bar{S}_2 - \bar{S}_1))$. This model is known as **Stommel's model**.

Since $\bar{S}_1 + \bar{S}_2$ is constant and the temperature difference is fixed, $\Delta T = T_2 - T_1 = 2 T^*$, the only variable remaining is the salinity difference, $\Delta S = \bar{S}_2 - \bar{S}_1$.

$\dfrac{d \Delta S}{d t} = \dfrac{d \bar{S}_2}{d t} - \dfrac{d \bar{S}_1}{d t} = 2 H + 2 |q| ( \bar{S}_1 - \bar{S}_2)$ 

#### Dynamical System

The problem is rendered dimensionless via non-dimensionalization by introducing the following linear scaling:

$x = \dfrac{\beta \Delta S}{\alpha \Delta T}, \ \ t' = 2 \alpha k | \Delta T|t, \ \ \lambda = \dfrac{\beta H}{\alpha^2 k \Delta T | \Delta T|}$  

The parameter $\lambda$ corresponds to a dimensionless surface salinity flux. Since $\Delta T = T_2 - T_1 > 0$ it follows that $\lambda > 0$.

The dynamics of the two-box model are now described by the scalar ordinary differential equation for the function $x: t \mapsto x(t)$ in $\mathbb{R}$,

$\dot{x} = \lambda - |1-x| x$

#### Equilibrium states

Equilibrium states are found by setting the right-hand side equal to zero.

$|1-x| x = \lambda$

Since $\lambda > 0$ any root $x^*$ is necessarily positive.

Solving the equation results in the following roots:
- $x^{*}_{1} = \dfrac{1}{2} + \dfrac{\sqrt{1 + 4 \lambda}}{2}$
	- Valid for all $\lambda > 0$
- $x^{*}_{2} = \dfrac{1}{2} + \dfrac{\sqrt{1 - 4 \lambda}}{2}$
	- Valid for $\lambda \in (0, \frac{1}{4})$
- $x^{*}_{3} = \dfrac{1}{2} - \dfrac{\sqrt{1 - 4 \lambda}}{2}$
	- Valid for $\lambda \in (0, \frac{1}{4})$
-  $x^{*}_{4} = \dfrac{1}{2} - \dfrac{\sqrt{1 - 4 \lambda}}{2}$
	- Not valid for any $\lambda$

Linearizing the equation around a critical point $x^*$ by substituting $x = x^*  + y$ results in the linearized equations:

$\dot{y} = \pm (2 x^* -1)y$

Values of $x > 1$ and $x < 1$ are distinguished due to the absolute value sign.

- $x^* = x^{*}_{1}$: 
	- Coefficient of $y$ is $- \sqrt{1 + 4 \lambda} < 0$  
	- Stable equilibrium point.
- $x^* = x^{*}_{2}$:
	- Coefficient of $y$ is $\sqrt{1- 4 \lambda} > 0$ 
	- Unstable equilibrium point.
- $x^* = x^{*}_{3}$:
	- Coefficient of $y$ is $- \sqrt{1 - 4 \lambda} < 0$
	- Stable again.

Reminder: $x = \dfrac{\beta \Delta S}{\alpha \Delta T}$, so what do the stable equilibrium points tell us about physical phenomena?
- $x^{*}_{1} > 1$ 
	- Salinity dominates the density differences between the two boxes.
	- This equilibrium is known as *S-mode*
	- In S-mode, overturning circulation is driven by salinity.
	- $q < 0 \implies$ surface flow directed towards equator, bottom flow directed towards poles.
- $x^{*}_{3} < 1$
	- Temperature is the dominant factor.
	- Equilibrium is known as *T-mode*.
	- $q > 0 \implies$ surface flow directed towards poles, bottom flow to equator.

#### Bifurcation

![[bifurcation_2box.jpg]]

- If system is initially at equilibrium on the upper branch (s-mode), it will remain in s-mode as the surface salinity flux increases/decreases.
- If the system is initially on the lower branch, it will remain in t-mode until $\lambda$ reaches a critical value of $\dfrac{1}{4}$ at which the system will transition to s-mode.
- Once in s-mode the system stays in s-mode and never transitions to t-mode.
- If negative values of $\lambda$ were allowed, it would be found that the system could jump from an upper to lower stable branch, creating a complete hysteresis loop. However, this would violate the laws of physics.