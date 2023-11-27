import sympy as sp
import matplotlib.pyplot as plt
import numpy as np

x, y  = sp.symbols('x y')
lambda_ = sp.symbols('lambda', real=True)

def jacobian(x1,y1,a):
    values = {x:x1, y:y1}
    jac = sp.Matrix([
        [-2*x*sp.sign(x - 1) - 2*sp.Abs(x - 1),
        -y*sp.sign(y - 1) - sp.Abs(y - 1)],
        [-x*sp.sign(x - 1) - sp.Abs(x - 1),
            -2*y*sp.sign(y - 1) - 2*sp.Abs(y - 1)]])
    jac = jac.subs(values)
    eigenvalues = jac.eigenvals()
    eigenvalues = list(eigenvalues.keys())
    if (eigenvalues[0].is_real and eigenvalues[1].is_real) and (eigenvalues[0] < 0 and eigenvalues[1] < 0):
        return True
    else:
        return False


x1 = 1/2 - sp.sqrt(9 + 12*lambda_)/6
x2 = 1/2 - sp.sqrt(9 - 12*lambda_)/6
x3 = 1/2 + sp.sqrt(9 - 12*lambda_)/6
x4 = 1/2 + sp.sqrt(9 + 12*lambda_)/6
y1 = 1/2 + sp.sqrt(9 + 12*lambda_)/6
y2 = 1/2 + sp.sqrt(9 - 12*lambda_)/6
y3 = 1/2 - sp.sqrt(9 + 12*lambda_)/6
y4 = 1/2 - sp.sqrt(9 - 12*lambda_)/6

lambdas = np.linspace(-2, 2, 100)

def x_tot(_, x, y):
     a = 2-x-y
     if not a.is_real:
          return None, None
     else:
          jac = jacobian(x,y,a)
          return a, jac

xs1, xs2, xs3, xs4, xs5, xs6, xs7, xs8, xs9, xs10, xs11, xs12, xs13, xs14, xs15, xs16=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]
xl1, xl2, xl3, xl4, xl5, xl6, xl7, xl8, xl9, xl10, xl11, xl12, xl13, xl14, xl15, xl16=[],[],[],[],[],[],[],[],[],[],[],[],[],[],[],[]

for i in lambdas:
     xs1.append(x_tot(i, x1.subs(lambda_, i), y1.subs(lambda_, i))[0])
     xl1.append(x_tot(i, x1.subs(lambda_, i), y1.subs(lambda_, i))[1])
     xs2.append(x_tot(i, x1.subs(lambda_, i), y2.subs(lambda_, i))[0])
     xl2.append(x_tot(i, x1.subs(lambda_, i), y2.subs(lambda_, i))[1])
     xs3.append(x_tot(i, x1.subs(lambda_, i), y3.subs(lambda_, i))[0])
     xl3.append(x_tot(i, x1.subs(lambda_, i), y3.subs(lambda_, i))[1])
     xs4.append(x_tot(i, x1.subs(lambda_, i), y4.subs(lambda_, i))[0])
     xl4.append(x_tot(i, x1.subs(lambda_, i), y4.subs(lambda_, i))[1])
     xs5.append(x_tot(i, x2.subs(lambda_, i), y1.subs(lambda_, i))[0])
     xl5.append(x_tot(i, x2.subs(lambda_, i), y1.subs(lambda_, i))[1])
     xs6.append(x_tot(i, x2.subs(lambda_, i), y2.subs(lambda_, i))[0])
     xl6.append(x_tot(i, x2.subs(lambda_, i), y2.subs(lambda_, i))[1])
     xs7.append(x_tot(i, x2.subs(lambda_, i), y3.subs(lambda_, i))[0])
     xl7.append(x_tot(i, x2.subs(lambda_, i), y3.subs(lambda_, i))[1])
     xs8.append(x_tot(i, x2.subs(lambda_, i), y4.subs(lambda_, i))[0])
     xl8.append(x_tot(i, x2.subs(lambda_, i), y4.subs(lambda_, i))[1])
     xs9.append(x_tot(i, x3.subs(lambda_, i), y1.subs(lambda_, i))[0])
     xl9.append(x_tot(i, x3.subs(lambda_, i), y1.subs(lambda_, i))[1])
     xs10.append(x_tot(i, x3.subs(lambda_, i), y2.subs(lambda_, i))[0])
     xl10.append(x_tot(i, x3.subs(lambda_, i), y2.subs(lambda_, i))[1])
     xs11.append(x_tot(i, x3.subs(lambda_, i), y3.subs(lambda_, i))[0])
     xl11.append(x_tot(i, x3.subs(lambda_, i), y3.subs(lambda_, i))[1])
     xs12.append(x_tot(i, x3.subs(lambda_, i), y4.subs(lambda_, i))[0])
     xl12.append(x_tot(i, x3.subs(lambda_, i), y4.subs(lambda_, i))[1])
     xs13.append(x_tot(i, x4.subs(lambda_, i), y1.subs(lambda_, i))[0])
     xl13.append(x_tot(i, x4.subs(lambda_, i), y1.subs(lambda_, i))[1])
     xs14.append(x_tot(i, x4.subs(lambda_, i), y2.subs(lambda_, i))[0])
     xl14.append(x_tot(i, x4.subs(lambda_, i), y2.subs(lambda_, i))[1])
     xs15.append(x_tot(i, x4.subs(lambda_, i), y3.subs(lambda_, i))[0])
     xl15.append(x_tot(i, x4.subs(lambda_, i), y3.subs(lambda_, i))[1])
     xs16.append(x_tot(i, x4.subs(lambda_, i), y4.subs(lambda_, i))[0])
     xl16.append(x_tot(i, x4.subs(lambda_, i), y4.subs(lambda_, i))[1])


# plt.figure(figsize=(10,10))

# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl1[i] is False], [xs1[i] for i in range(len(lambdas)) if xl1[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl1[i] is True], [xs1[i] for i in range(len(lambdas)) if xl1[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl2[i] is False], [xs2[i] for i in range(len(lambdas)) if xl2[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl2[i] is True], [xs2[i] for i in range(len(lambdas)) if xl2[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl3[i] is False], [xs3[i] for i in range(len(lambdas)) if xl3[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl3[i] is True], [xs3[i] for i in range(len(lambdas)) if xl3[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl4[i] is False], [xs4[i] for i in range(len(lambdas)) if xl4[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl4[i] is True], [xs4[i] for i in range(len(lambdas)) if xl4[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl5[i] is False], [xs5[i] for i in range(len(lambdas)) if xl5[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl5[i] is True], [xs5[i] for i in range(len(lambdas)) if xl5[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl6[i] is False], [xs6[i] for i in range(len(lambdas)) if xl6[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl6[i] is True], [xs6[i] for i in range(len(lambdas)) if xl6[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl7[i] is False], [xs7[i] for i in range(len(lambdas)) if xl7[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl7[i] is True], [xs7[i] for i in range(len(lambdas)) if xl7[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl8[i] is False], [xs8[i] for i in range(len(lambdas)) if xl8[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl8[i] is True], [xs8[i] for i in range(len(lambdas)) if xl8[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl9[i] is False], [xs9[i] for i in range(len(lambdas)) if xl9[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl9[i] is True], [xs9[i] for i in range(len(lambdas)) if xl9[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl10[i] is False], [xs10[i] for i in range(len(lambdas)) if xl10[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl10[i] is True], [xs10[i] for i in range(len(lambdas)) if xl10[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl11[i] is False], [xs11[i] for i in range(len(lambdas)) if xl11[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl11[i] is True], [xs11[i] for i in range(len(lambdas)) if xl11[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl12[i] is False], [xs12[i] for i in range(len(lambdas)) if xl12[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl12[i] is True], [xs12[i] for i in range(len(lambdas)) if xl12[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl13[i] is False], [xs13[i] for i in range(len(lambdas)) if xl13[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl13[i] is True], [xs13[i] for i in range(len(lambdas)) if xl13[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl14[i] is False], [xs14[i] for i in range(len(lambdas)) if xl14[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl14[i] is True], [xs14[i] for i in range(len(lambdas)) if xl14[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl15[i] is False], [xs15[i] for i in range(len(lambdas)) if xl15[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl15[i] is True], [xs15[i] for i in range(len(lambdas)) if xl15[i] is True])
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl16[i] is False], [xs16[i] for i in range(len(lambdas)) if xl16[i] is False], linestyle='--')
# plt.plot([lambdas[i] for i in range(len(lambdas)) if xl16[i] is True], [xs16[i] for i in range(len(lambdas)) if xl16[i] is True])

# plt.axhline(y=0, color='black', linestyle='--')

# # increase font size 
# plt.rcParams.update({'font.size': 16})
# plt.xlabel(r'$\lambda$')
# plt.ylabel(r'$2-x-y$')
# plt.xticks(np.arange(-1, 1.5, 0.5))
# plt.show()



plt.figure(figsize=(10,10))

plt.plot([lambdas[i] for i in range(len(lambdas)) if xl1[i] is False], [xs1[i] for i in range(len(lambdas)) if xl1[i] is False], linestyle='--', color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl1[i] is True], [xs1[i] for i in range(len(lambdas)) if xl1[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl2[i] is False], [xs2[i] for i in range(len(lambdas)) if xl2[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl2[i] is True], [xs2[i] for i in range(len(lambdas)) if xl2[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl3[i] is False], [xs3[i] for i in range(len(lambdas)) if xl3[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl3[i] is True], [xs3[i] for i in range(len(lambdas)) if xl3[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl4[i] is False], [xs4[i] for i in range(len(lambdas)) if xl4[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl4[i] is True], [xs4[i] for i in range(len(lambdas)) if xl4[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl5[i] is False], [xs5[i] for i in range(len(lambdas)) if xl5[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl5[i] is True], [xs5[i] for i in range(len(lambdas)) if xl5[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl6[i] is False], [xs6[i] for i in range(len(lambdas)) if xl6[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl6[i] is True], [xs6[i] for i in range(len(lambdas)) if xl6[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl7[i] is False], [xs7[i] for i in range(len(lambdas)) if xl7[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl7[i] is True], [xs7[i] for i in range(len(lambdas)) if xl7[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl8[i] is False], [xs8[i] for i in range(len(lambdas)) if xl8[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl8[i] is True], [xs8[i] for i in range(len(lambdas)) if xl8[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl9[i] is False], [xs9[i] for i in range(len(lambdas)) if xl9[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl9[i] is True], [xs9[i] for i in range(len(lambdas)) if xl9[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl10[i] is False], [xs10[i] for i in range(len(lambdas)) if xl10[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl10[i] is True], [xs10[i] for i in range(len(lambdas)) if xl10[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl11[i] is False], [xs11[i] for i in range(len(lambdas)) if xl11[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl11[i] is True], [xs11[i] for i in range(len(lambdas)) if xl11[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl12[i] is False], [xs12[i] for i in range(len(lambdas)) if xl12[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl12[i] is True], [xs12[i] for i in range(len(lambdas)) if xl12[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl13[i] is False], [xs13[i] for i in range(len(lambdas)) if xl13[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl13[i] is True], [xs13[i] for i in range(len(lambdas)) if xl13[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl14[i] is False], [xs14[i] for i in range(len(lambdas)) if xl14[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl14[i] is True], [xs14[i] for i in range(len(lambdas)) if xl14[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl15[i] is False], [xs15[i] for i in range(len(lambdas)) if xl15[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl15[i] is True], [xs15[i] for i in range(len(lambdas)) if xl15[i] is True],color = 'blue', alpha=0.5)
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl16[i] is False], [xs16[i] for i in range(len(lambdas)) if xl16[i] is False], linestyle='--',color = 'red')
plt.plot([lambdas[i] for i in range(len(lambdas)) if xl16[i] is True], [xs16[i] for i in range(len(lambdas)) if xl16[i] is True],color = 'blue', alpha=0.5)

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
plt.xticks([])

# plt.gca().set_aspect('equal', adjustable='box')

plt.rcParams['figure.figsize'] = [8, 10]
plt.rcParams['figure.dpi'] = 600

# plt.xticks([-0.75, 0.75])
plt.show()

# plt.savefig('three_box_bifurcation_diagram.png', dpi=600)