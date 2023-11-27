import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

x, y  = sp.symbols('x y')
l = sp.symbols('lambda', real=True)

def jacobian(x1,y1):
    values = {x:x1, y:y1}
    jac = sp.Matrix([
        [-2*x*sp.sign(x - 1) - 2*sp.Abs(x - 1),
        -y*sp.sign(y - 1) - sp.Abs(y - 1)],
        [-x*sp.sign(x - 1) - sp.Abs(x - 1),
            -2*y*sp.sign(y - 1) - 2*sp.Abs(y - 1)]])
    return jac.subs(values)

x1 = 1/2 - sp.sqrt(9 + 12*l)/6
x2 = 1/2 - sp.sqrt(9 - 12*l)/6
x3 = 1/2 + sp.sqrt(9 - 12*l)/6
x4 = 1/2 + sp.sqrt(9 + 12*l)/6
y1 = 1/2 + sp.sqrt(9 + 12*l)/6
y2 = 1/2 + sp.sqrt(9 - 12*l)/6
y3 = 1/2 - sp.sqrt(9 + 12*l)/6
y4 = 1/2 - sp.sqrt(9 - 12*l)/6

lambdas = np.linspace(-2, 2, 1000)
min_lambda, max_lambda = -0.75, 0.75

def stable(X):
    if X is None:
        return None
    else:
        j = jacobian(X[0], X[1])
        eigenvals = [sp.re(i) for i in j.eigenvals().keys()]
        if eigenvals[0] < 0 and eigenvals[1] < 0:
            return 1 # Stable
        elif (eigenvals[0] < 0 and eigenvals[1] > 0) or (eigenvals[0] > 0 and eigenvals[1] < 0):
            return 2 # Saddle
        else:
            return 3 # Unstable
    
# 16 different equlibria

x1y1 = [(x1.subs(l, lambda_), y1.subs(l, lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
stablex1y1 = [stable(i) for i in x1y1]
x1y2 = [(x1.subs(l, lambda_), y2.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex1y2 = [stable(i) for i in x1y2]
x1y3 = [(x1.subs(l, lambda_), y3.subs(l, lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
stablex1y3 = [stable(i) for i in x1y3]
x1y4 = [(x1.subs(l, lambda_), y4.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex1y4 = [stable(i) for i in x1y4]
x2y1 = [(x2.subs(l, lambda_), y1.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex2y1 = [stable(i) for i in x2y1]
x2y2 = [(x2.subs(l, lambda_), y2.subs(l, lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex2y2 = [stable(i) for i in x2y2]
x2y3 = [(x2.subs(l, lambda_), y3.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex2y3 = [stable(i) for i in x2y3]
x2y4 = [(x2.subs(l, lambda_), y4.subs(l, lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex2y4 = [stable(i) for i in x2y4]
x3y1 = [(x3.subs(l, lambda_), y1.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas] 
stablex3y1 = [stable(i) for i in x3y1]
x3y2 = [(x3.subs(l, lambda_), y2.subs(l, lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex3y2 = [stable(i) for i in x3y2]
x3y3 = [(x3.subs(l, lambda_), y3.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex3y3 = [stable(i) for i in x3y3]
x3y4 = [(x3.subs(l, lambda_), y4.subs(l, lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex3y4 = [stable(i) for i in x3y4]
x4y1 = [(x4.subs(l, lambda_), y1.subs(l, lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
stablex4y1 = [stable(i) for i in x4y1]
x4y2 = [(x4.subs(l, lambda_), y2.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex4y2 = [stable(i) for i in x4y2]
x4y3 = [(x4.subs(l, lambda_), y3.subs(l, lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
stablex4y3 = [stable(i) for i in x4y3]
x4y4 = [(x4.subs(l, lambda_), y4.subs(l, lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
stablex4y4 = [stable(i) for i in x4y4]

all_equilibria = [x1y1, x1y2, x1y3, x1y4, x2y1, x2y2, x2y3, x2y4, x3y1, x3y2, x3y3, x3y4, x4y1, x4y2, x4y3, x4y4]
all_stability = [stablex1y1, stablex1y2, stablex1y3, stablex1y4, stablex2y1, stablex2y2, stablex2y3, stablex2y4, stablex3y1, stablex3y2, stablex3y3, stablex3y4, stablex4y1, stablex4y2, stablex4y3, stablex4y4]

plt.figure(figsize=(8,10))

for i, l in enumerate(lambdas):
    if i%100 == 0:
        print(i)
    for j, eq in enumerate(all_equilibria):
        if eq[i] is not None:
            if all_stability[j][i] == 1:
                plt.scatter(l, 2-eq[i][0]-eq[i][1], c='b', alpha=0.1)
            elif all_stability[j][i] == 2:
                plt.scatter(l, 2-eq[i][0]-eq[i][1], facecolors='none', edgecolors='g', alpha=0.2)
            else:
                plt.scatter(l, 2-eq[i][0]-eq[i][1], facecolors='none', edgecolors='r', alpha=0.2)
                continue

# Adding grid lines, especially vertical lines at x +/- 3/4
plt.axvline(x=-3/4, color='gray', linewidth=0.5, linestyle='--')
plt.axvline(x=3/4, color='gray', linewidth=0.5, linestyle='--')
plt.axhline(y=1, color='gray', linewidth=0.5, linestyle='--')
plt.axhline(y=2, color='gray', linewidth=0.5, linestyle='--')
plt.axhline(y=-1, color='gray', linewidth=0.5, linestyle='--')

plt.gca().xaxis.set_ticks_position('bottom')
plt.gca().yaxis.set_ticks_position('left')

plt.gca().spines['right'].set_color('none')
plt.gca().spines['top'].set_color('none')
plt.gca().spines['bottom'].set_position('zero')
plt.gca().spines['left'].set_position('zero')


# Remove the x-tick at the origin (0)
# xticks = plt.xticks()[0]
# xticks = [tick for tick in xticks if tick != 0]
# plt.xticks(xticks)

plt.yticks([])

# plt.gca().set_aspect('equal', adjustable='box')

plt.rcParams['figure.figsize'] = [10, 10]
plt.rcParams['figure.dpi'] = 600




plt.xticks([-0.75, 0.75])
# plt.show()

plt.savefig('bifurcation_diagram.png', dpi=600)