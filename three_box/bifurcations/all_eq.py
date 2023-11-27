import matplotlib.pyplot as plt
import numpy as np
from phaseportrait import PhasePortrait2D
from math import sqrt

"""
Simple script producing a sequence of phase portraits, along with equilibrium points (marked in red) 
as a function of the parameter lambda (salinity flux)
"""
# defining all possible equilibrium points (16 in total)

x1 = lambda lambda_: 1/2 - np.sqrt(9 + 12*lambda_)/6
x2 = lambda lambda_: 1/2 - np.sqrt(9 - 12*lambda_)/6
x3 = lambda lambda_: 1/2 + np.sqrt(9 - 12*lambda_)/6
x4 = lambda lambda_: 1/2 + np.sqrt(9 + 12*lambda_)/6

y1 = lambda lambda_: 1/2 + np.sqrt(9 + 12*lambda_)/6
y2 = lambda lambda_: 1/2 + np.sqrt(9 - 12*lambda_)/6
y3 = lambda lambda_: 1/2 - np.sqrt(9 + 12*lambda_)/6
y4 = lambda lambda_: 1/2 - np.sqrt(9 - 12*lambda_)/6

lambdas = np.linspace(-2, 2, 20)


# Defining dF function

def three_box_dF(x, y, lambda_):
    dx = lambda_ - y*abs(1-y) - 2*x*abs(1-x)
    dy = lambda_ - x*abs(1-x) - 2*y*abs(1-y)
    return dx, dy

min_lambda, max_lambda = -3/4, 3/4


def stable(lst):
    return [True if lst[i] != None and lst[i][0] < 0 and lst[i][1] < 0 else False for i in range(len(lst))]
# List of all equilibrium points for each lambda (if real)
# List of stability of each equilibrium point for each lambda (if real) - True = stable, False = unstable

x1y1 = [(x1(lambda_), y1(lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
eigenvalsx1y1 = [None, None, None, None, None, None, [-0.2294156927, 0.2294156927], [-0.9459053027, 0.9459053027], [-1.317893060, 1.317893060], [-1.605910133, 1.605910133], [-1.067872127, -3.203616384], [-1.192079120, -3.576237360], [-1.304513084, -3.913539252], [-1.407997212, -4.223991636], [-1.504379572, -4.513138716], [-1.594948166, -4.784844498], [-5.041929451, -1.680643151], [-1.762175690, -5.286527069], [-5.520297468, -1.840099156], [-sqrt(33)/3, -sqrt(33)]]
stablex1y1 = stable(eigenvalsx1y1)

x1y2 = [(x1(lambda_), y2(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx1y2 = [None, None, None, None, None, None, [-0.1937855624, -2.887115285], [-0.7159154856, -2.985348044], [-0.9075159468, -2.998414119], [-0.9901078761, -2.999981676], [-1.869890570, 1.588491611], [-2.136196493, 1.273810080], [-2.405336711, 0.8885479048], [-2.754223733, 0.2031357338], None, None, None, None, None, None]
stablex1y2 = stable(eigenvalsx1y2)


x1y3 = [(x1(lambda_), y3(lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
eigenvalsx1y3 = [None, None, None, None, None, None, [-0.1324532120, -0.3973596356], [-0.5461186814, -1.638356043], [-0.7608859126, -2.282657739], [-0.9271726476, -2.781517944], [-1.067872127, -3.203616383], [-1.192079119, -3.576237359], [-1.304513084, -3.913539252], [-1.407997211, -4.223991635], [-1.504379571, -4.513138715], [-1.594948166, -4.784844498], [-1.680643151, -5.041929451], [-1.762175689, -5.286527069], [-1.840099157, -5.520297469], [-sqrt(33)/3, -sqrt(33)]]
stablex1y3 = stable(eigenvalsx1y3)

x1y4 = [(x1(lambda_), y4(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx1y4 = [None, None, None, None, None, None, [-0.1937855624, -2.887115284], [-0.7159154856, -2.985348044], [-0.9075159465, -2.998414118], [-0.9901078757, -2.999981675], [-2.999981675, -0.9901078757], [-2.998414118, -0.9075159465], [-2.985348044, -0.7159154856], [-2.887115284, -0.1937855624], None, None, None, None, None, None]
stablex1y4 = stable(eigenvalsx1y4)


x2y1 = [(x2(lambda_), y1(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx2y1 = [None, None, None, None, None, None, [-2.754223733, 0.2031357338], [-2.405336711, 0.8885479048], [-2.136196493, 1.273810080], [-1.869890570, 1.588491611], [-0.9901078761, -2.999981676], [-0.9075159468, -2.998414119], [-0.7159154856, -2.985348044], [-0.1937855624, -2.887115285], None, None, None, None, None, None]
stablex2y1 = stable(eigenvalsx2y1)

x2y2 = [(x2(lambda_), y2(lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx2y2 = [[-sqrt(33)/3, -sqrt(33)], [-5.520297468, -1.840099156], [-1.762175690, -5.286527069], [-5.041929451, -1.680643151], [-1.594948166, -4.784844498], [-1.504379572, -4.513138716], [-1.407997212, -4.223991636], [-1.304513084, -3.913539252], [-1.192079120, -3.576237360], [-1.067872127, -3.203616384], [-1.605910133, 1.605910133], [-1.317893060, 1.317893060], [-0.9459053027, 0.9459053027], [-0.2294156927, 0.2294156927], None, None, None, None, None, None]
stablex2y2 = stable(eigenvalsx2y2)


x2y3 = [(x2(lambda_), y3(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx2y3 = [None, None, None, None, None, None, [-2.887115284, -0.1937855624], [-2.985348044, -0.7159154856], [-2.998414118, -0.9075159465], [-2.999981675, -0.9901078757], [-0.9901078757, -2.999981675], [-0.9075159465, -2.998414118], [-0.7159154856, -2.985348044], [-0.1937855624, -2.887115284], None, None, None, None, None, None]
stablex2y3 = stable(eigenvalsx2y3)


x2y4 = [(x2(lambda_), y4(lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx2y4 = [[-sqrt(33)/3, -sqrt(33)], [-1.840099157, -5.520297469], [-1.762175689, -5.286527069], [-1.680643151, -5.041929451], [-1.594948166, -4.784844498], [-1.504379571, -4.513138715], [-1.407997211, -4.223991635], [-1.304513084, -3.913539252], [-1.192079119, -3.576237359], [-1.067872127, -3.203616383], [-0.9271726476, -2.781517944], [-0.7608859126, -2.282657739], [-0.5461186814, -1.638356043], [-0.1324532120, -0.3973596356], None, None, None, None, None, None]
stablex2y4 = stable(eigenvalsx2y4)


x3y1 = [(x3(lambda_), y1(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas] 
eigenvalsx3y1 = [None, None, None, None, None, None, [-2.754223734, 0.2031357338], [-2.405336711, 0.8885479048], [-2.136196494, 1.273810080], [-1.869890571, 1.588491611], [1.588491611, -1.869890571], [1.273810080, -2.136196494], [0.8885479048, -2.405336711], [0.2031357338, -2.754223734], None, None, None, None, None, None]
stablex3y1 = stable(eigenvalsx3y1)


x3y2 = [(x3(lambda_), y2(lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx3y2 = [[-sqrt(33)/3, -sqrt(33)], [-1.840099156, -5.520297468], [-1.762175690, -5.286527070], [-1.680643150, -5.041929450], [-1.594948166, -4.784844498], [-1.504379572, -4.513138716], [-1.407997212, -4.223991636], [-1.304513084, -3.913539252], [-1.192079120, -3.576237360], [-1.067872128, -3.203616384], [2.781517944, 0.9271726476], [2.282657739, 0.7608859126], [1.638356043, 0.5461186814], [0.3973596356, 0.1324532120], None, None, None, None, None, None]
stablex3y2 = stable(eigenvalsx3y2)



x3y3 = [(x3(lambda_), y3(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx3y3 = [None, None, None, None, None, None, [-2.887115285, -0.1937855624], [-2.985348044, -0.7159154856], [-2.998414119, -0.9075159468], [-2.999981676, -0.9901078761], [1.588491611, -1.869890570], [1.273810080, -2.136196493], [0.8885479048, -2.405336711], [0.2031357338, -2.754223733], None, None, None, None, None, None]
stablex3y3 = stable(eigenvalsx3y3)


x3y4 = [(x3(lambda_), y4(lambda_)) if lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx3y4 = [[-sqrt(33)/3, -sqrt(33)], [-1.840099157, -5.520297468], [-5.286527069, -1.762175690], [-1.680643150, -5.041929450], [-1.594948166, -4.784844498], [-4.513138715, -1.504379571], [-4.223991635, -1.407997211], [-1.304513084, -3.913539252], [-3.576237359, -1.192079119], [-3.203616384, -1.067872128], [1.605910133, -1.605910133], [1.317893060, -1.317893060], [0.9459053027, -0.9459053027], [0.2294156927, -0.2294156927], None, None, None, None, None, None]
stablex3y4 = stable(eigenvalsx3y4)


x4y1 = [(x4(lambda_), y1(lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
eigenvalsx4y1 = [None, None, None, None, None, None, [0.3973596356, 0.1324532120], [1.638356043, 0.5461186814], [2.282657739, 0.7608859126], [2.781517944, 0.9271726476], [-1.067872128, -3.203616384], [-1.192079120, -3.576237360], [-1.304513084, -3.913539252], [-1.407997212, -4.223991636], [-1.504379572, -4.513138716], [-1.594948166, -4.784844498], [-1.680643150, -5.041929450], [-1.762175690, -5.286527070], [-1.840099156, -5.520297468], [-sqrt(33)/3, -sqrt(33)]]
stablex4y1 = stable(eigenvalsx4y1)


x4y2 = [(x4(lambda_), y2(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx4y2 = [None, None, None, None, None, None, [0.2031357338, -2.754223734], [0.8885479048, -2.405336711], [1.273810080, -2.136196494], [1.588491611, -1.869890571], [-1.869890571, 1.588491611], [-2.136196494, 1.273810080], [-2.405336711, 0.8885479048], [-2.754223734, 0.2031357338], None, None, None, None, None, None]
stablex4y2 = stable(eigenvalsx4y2)


x4y3 = [(x4(lambda_), y3(lambda_)) if lambda_ >= min_lambda else None for lambda_ in lambdas]
eigenvalsx4y3 = [None, None, None, None, None, None, [0.2294156927, -0.2294156927], [0.9459053027, -0.9459053027], [1.317893060, -1.317893060], [1.605910133, -1.605910133], [-3.203616384, -1.067872128], [-3.576237359, -1.192079119], [-1.304513084, -3.913539252], [-4.223991635, -1.407997211], [-4.513138715, -1.504379571], [-1.594948166, -4.784844498], [-1.680643150, -5.041929450], [-5.286527069, -1.762175690], [-1.840099157, -5.520297468], [-sqrt(33)/3, -sqrt(33)]]
stablex4y3 = stable(eigenvalsx4y3)


x4y4 = [(x4(lambda_), y4(lambda_)) if lambda_ >= min_lambda and lambda_ <= max_lambda else None for lambda_ in lambdas]
eigenvalsx4y4 = [None, None, None, None, None, None, [0.2031357338, -2.754223733], [0.8885479048, -2.405336711], [1.273810080, -2.136196493], [1.588491611, -1.869890570], [-2.999981676, -0.9901078761], [-2.998414119, -0.9075159468], [-2.985348044, -0.7159154856], [-2.887115285, -0.1937855624], None, None, None, None, None, None]
stablex4y4 = stable(eigenvalsx4y4)


all_equilibria = [x1y1, x1y2, x1y3, x1y4, x2y1, x2y2, x2y3, x2y4, x3y1, x3y2, x3y3, x3y4, x4y1, x4y2, x4y3, x4y4]
all_stability = [stablex1y1, stablex1y2, stablex1y3, stablex1y4, stablex2y1, stablex2y2, stablex2y3, stablex2y4, stablex3y1, stablex3y2, stablex3y3, stablex3y4, stablex4y1, stablex4y2, stablex4y3, stablex4y4]

# First plot of just the equilibrium points

colors = ['red', 'green', 'blue', 'yellow', 'magenta', 'cyan', 'orange', 'purple', 'darkgreen', 'navy', 'gold', 'saddlebrown', 'indigo', 'brown', 'tomato', 'steelblue']



# Function that plots all equilibrium points and their stability for a given lambda
def plot_equilibria(x1y1=x1y1, x1y2=x1y2, x1y3=x1y3, x1y4=x1y4, x2y1=x2y1, x2y2=x2y2, x2y3=x2y3, x2y4=x2y4, x3y1=x3y1, x3y2=x3y2, x3y3=x3y3, x3y4=x3y4, x4y1=x4y1, x4y2=x4y2, x4y3=x4y3, x4y4=x4y4, lambdas=lambdas,colors=colors):
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlim(-2, 2)
    ax.set_ylim(-2, 2)
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    

    # same as above, but now filling in the stable equilibria and leaving the unstable equilibria unfilled

    for i in range(len(lambdas)):
        if x1y1[i] != None:
            if stablex1y1[i] == True:
                ax.scatter(x1y1[i][0], x1y1[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x1y1[i][0], x1y1[i][1],  s=50, facecolors='none', color='red')
        if x1y2[i] != None:
            if stablex1y2[i] == True:
                ax.scatter(x1y2[i][0], x1y2[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x1y2[i][0], x1y2[i][1],  s=50, facecolors='none', color='red')
        if x1y3[i] != None:
            if stablex1y3[i] == True:
                ax.scatter(x1y3[i][0], x1y3[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x1y3[i][0], x1y3[i][1],  s=50, facecolors='none', color='red')
        if x1y4[i] != None:
            if stablex1y4[i] == True:
                ax.scatter(x1y4[i][0], x1y4[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x1y4[i][0], x1y4[i][1],  s=50, facecolors='none', color='red')
        if x2y1[i] != None:
            if stablex2y1[i] == True:
                ax.scatter(x2y1[i][0], x2y1[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x2y1[i][0], x2y1[i][1],  s=50, facecolors='none', color='red')
        if x2y2[i] != None:
            if stablex2y2[i] == True:
                ax.scatter(x2y2[i][0], x2y2[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x2y2[i][0], x2y2[i][1],  s=50, facecolors='none', color='red')
        if x2y3[i] != None:
            if stablex2y3[i] == True:
                ax.scatter(x2y3[i][0], x2y3[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x2y3[i][0], x2y3[i][1],  s=50, facecolors='none', color='red')
        if x2y4[i] != None:
            if stablex2y4[i] == True:
                ax.scatter(x2y4[i][0], x2y4[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x2y4[i][0], x2y4[i][1],  s=50, facecolors='none', color='red')
        if x3y1[i] != None:
            if stablex3y1[i] == True:
                ax.scatter(x3y1[i][0], x3y1[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x3y1[i][0], x3y1[i][1],  s=50, facecolors='none', color='red')
        if x3y2[i] != None:
            if stablex3y2[i] == True:
                ax.scatter(x3y2[i][0], x3y2[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x3y2[i][0], x3y2[i][1],  s=50, facecolors='none', color='red')
        if x3y3[i] != None:
            if stablex3y3[i] == True:
                ax.scatter(x3y3[i][0], x3y3[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x3y3[i][0], x3y3[i][1],  s=50, facecolors='none', color='red')
        if x3y4[i] != None:
            if stablex3y4[i] == True:
                ax.scatter(x3y4[i][0], x3y4[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x3y4[i][0], x3y4[i][1],  s=50, facecolors='none', color='red')
        if x4y1[i] != None:
            if stablex4y1[i] == True:
                ax.scatter(x4y1[i][0], x4y1[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x4y1[i][0], x4y1[i][1],  s=50, facecolors='none', color='red')
        if x4y2[i] != None:
            if stablex4y2[i] == True:
                ax.scatter(x4y2[i][0], x4y2[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x4y2[i][0], x4y2[i][1],  s=50, facecolors='none', color='red')
        if x4y3[i] != None:
            if stablex4y3[i] == True:
                ax.scatter(x4y3[i][0], x4y3[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x4y3[i][0], x4y3[i][1],  s=50, facecolors='none', color='red')
        if x4y4[i] != None:
            if stablex4y4[i] == True:
                ax.scatter(x4y4[i][0], x4y4[i][1],  s=50, alpha=0.5, color='blue')
            else:
                ax.scatter(x4y4[i][0], x4y4[i][1],  s=50, facecolors='none', color='red')

    plt.grid()
    ax.set_ylim(-0.6, 1.6)
    ax.set_xlim(-0.6, 1.6)

    plt.show()


def plot_eigenvalues(eigenvals, lambdas=lambdas):
    # Same as above function, but now plotting the eigenvalues of each equilibrium point as a function of lambda
    fig, ax = plt.subplots(figsize=(10, 10))

    for i, l in enumerate(lambdas): # Iterate through lambdas
        if eigenvals[i] != None:
            for j in eigenvals[i]:  # iterate through eigenvals
                if j >= 0: # Check stability
                    ax.scatter(l, j, s=50, color='red', facecolors='none')
                else:
                    ax.scatter(l, j, s=50, color='blue')
                
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'Eigenvalues')
    plt.grid()
    plt.show()


# Making a 3D plot of equilibrium points as a function of lambda

def plot_eq_3D(x1y1=x1y1, x1y2=x1y2, x1y3=x1y3, x1y4=x1y4, x2y1=x2y1, x2y2=x2y2, x2y3=x2y3, x2y4=x2y4, x3y1=x3y1, x3y2=x3y2, x3y3=x3y3, x3y4=x3y4, x4y1=x4y1, x4y2=x4y2, x4y3=x4y3, x4y4=x4y4, lambdas=lambdas):

    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel(r'lambda')
    ax.set_ylabel('x')
    ax.set_zlabel('y')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_zlim(-0.6, 1.6)

    for i in range(len(lambdas)):
        if x1y1[i] != None:
            if stablex1y1[i] == True:
                ax.scatter(lambdas[i], x1y1[i][0], x1y1[i][1], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x1y1[i][0], x1y1[i][1], s=50, color='red', facecolors='none')
        if x1y2[i] != None:
            if stablex1y2[i] == True:
                ax.scatter(lambdas[i], x1y2[i][0], x1y2[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x1y2[i][0], x1y2[i][1], s=50, color='red', facecolors='none')
        if x1y3[i] != None:
            if stablex1y3[i] == True:
                ax.scatter(lambdas[i], x1y3[i][0], x1y3[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x1y3[i][0], x1y3[i][1], s=50, color='red', facecolors='none')
        if x1y4[i] != None:
            if stablex1y4[i] == True:
                ax.scatter(lambdas[i], x1y4[i][0], x1y4[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x1y4[i][0], x1y4[i][1], s=50, color='red', facecolors='none')
        if x2y1[i] != None:
            if stablex2y1[i] == True:
                ax.scatter(lambdas[i], x2y1[i][0], x2y1[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x2y1[i][0], x2y1[i][1], s=50, color='red', facecolors='none')
        if x2y2[i] != None:
            if stablex2y2[i] == True:
                ax.scatter(lambdas[i], x2y2[i][0], x2y2[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x2y2[i][0], x2y2[i][1], s=50, color='red', facecolors='none')
        if x2y3[i] != None:
            if stablex2y3[i] == True:
                ax.scatter(lambdas[i], x2y3[i][0], x2y3[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x2y3[i][0], x2y3[i][1], s=50, color='red', facecolors='none')
        if x2y4[i] != None:
            if stablex2y4[i] == True:
                ax.scatter(lambdas[i], x2y4[i][0], x2y4[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x2y4[i][0], x2y4[i][1], s=50, color='red', facecolors='none')
        if x3y1[i] != None:
            if stablex3y1[i] == True:
                ax.scatter(lambdas[i], x3y1[i][0], x3y1[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x3y1[i][0], x3y1[i][1], s=50, color='red', facecolors='none')
        if x3y2[i] != None:
            if stablex3y2[i] == True:
                ax.scatter(lambdas[i], x3y2[i][0], x3y2[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x3y2[i][0], x3y2[i][1], s=50, color='red', facecolors='none')
        if x3y3[i] != None:
            if stablex3y3[i] == True:
                ax.scatter(lambdas[i], x3y3[i][0], x3y3[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x3y3[i][0], x3y3[i][1], s=50, color='red', facecolors='none')
        if x3y4[i] != None:
            if stablex3y4[i] == True:
                ax.scatter(lambdas[i], x3y4[i][0], x3y4[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x3y4[i][0], x3y4[i][1], s=50, color='red', facecolors='none')
        if x4y1[i] != None:
            if stablex4y1[i] == True:
                ax.scatter(lambdas[i], x4y1[i][0], x4y1[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x4y1[i][0], x4y1[i][1], s=50, color='red', facecolors='none')
        if x4y2[i] != None:
            if stablex4y2[i] == True:
                ax.scatter(lambdas[i], x4y2[i][0], x4y2[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x4y2[i][0], x4y2[i][1], s=50, color='red', facecolors='none')
        if x4y3[i] != None:
            if stablex4y3[i] == True:
                ax.scatter(lambdas[i], x4y3[i][0], x4y3[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x4y3[i][0], x4y3[i][1], s=50, color='red', facecolors='none')
        if x4y4[i] != None:
            if stablex4y4[i] == True:
                ax.scatter(lambdas[i], x4y4[i][0], x4y4[i][1], s=50, color = 'blue', alpha=0.5)
            else:
                ax.scatter(lambdas[i], x4y4[i][0], x4y4[i][1], s=50, color='red', facecolors='none')
        
    plt.show()




def eq_3D_lam_z_axis():
    # Same as above, but with lambda as z-axis
    fig = plt.figure(figsize=(10, 10))
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlabel('x')
    ax.set_ylabel('y')
    ax.set_zlabel(r'$\lambda$')
    ax.set_xlim(-2, 2)
    ax.set_ylim(-0.6, 1.6)
    ax.set_zlim(-2, 2)

    for i in range(len(lambdas)):

        if x1y1[i] != None:
            if stablex1y1[i] == True:
                ax.scatter(x1y1[i][0], x1y1[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x1y1[i][0], x1y1[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x1y2[i] != None:
            if stablex1y2[i] == True:
                ax.scatter(x1y2[i][0], x1y2[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x1y2[i][0], x1y2[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x1y3[i] != None:
            if stablex1y3[i] == True:
                ax.scatter(x1y3[i][0], x1y3[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x1y3[i][0], x1y3[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x1y4[i] != None:
            if stablex1y4[i] == True:
                ax.scatter(x1y4[i][0], x1y4[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x1y4[i][0], x1y4[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x2y1[i] != None:
            if stablex2y1[i] == True:
                ax.scatter(x2y1[i][0], x2y1[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x2y1[i][0], x2y1[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x2y2[i] != None:
            if stablex2y2[i] == True:
                ax.scatter(x2y2[i][0], x2y2[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x2y2[i][0], x2y2[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x2y3[i] != None:
            if stablex2y3[i] == True:
                ax.scatter(x2y3[i][0], x2y3[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x2y3[i][0], x2y3[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x2y4[i] != None:
            if stablex2y4[i] == True:
                ax.scatter(x2y4[i][0], x2y4[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x2y4[i][0], x2y4[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x3y1[i] != None:
            if stablex3y1[i] == True:
                ax.scatter(x3y1[i][0], x3y1[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x3y1[i][0], x3y1[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x3y2[i] != None:
            if stablex3y2[i] == True:
                ax.scatter(x3y2[i][0], x3y2[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x3y2[i][0], x3y2[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x3y3[i] != None:
            if stablex3y3[i] == True:
                ax.scatter(x3y3[i][0], x3y3[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x3y3[i][0], x3y3[i][1], lambdas[i], s=50, color='red', facecolors='none')
        if x3y4[i] != None:
            if stablex3y4[i] == True:
                ax.scatter(x3y4[i][0], x3y4[i][1], lambdas[i], s=50, color='blue', alpha=0.5)
            else:
                ax.scatter(x3y4[i][0], x3y4[i][1], lambdas[i], s=50, color='red', facecolors='none')


def plot_X_eq():
    # Function that just plots x-coordinates of equilibrium points as a function of lambda, and distinguishes between stable and unstable equilibria

    fig, ax = plt.subplots(figsize=(10, 10))
    # plt.rcParams.update({'dpi': 600})
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$x$')
    # ax.set_xlim(-2, 2)

    for i, l in enumerate(lambdas):
        if x1y1[i] != None:
            if stablex1y1[i] == True:
                ax.scatter(l, x1y1[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x1y1[i][0], s=50, color='red', facecolors='none')
        if x1y2[i] != None:
            if stablex1y2[i] == True:
                ax.scatter(l, x1y2[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x1y2[i][0], s=50, color='red', facecolors='none')
        if x1y3[i] != None:
            if stablex1y3[i] == True:
                ax.scatter(l, x1y3[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x1y3[i][0], s=50, color='red', facecolors='none')
        if x1y4[i] != None:
            if stablex1y4[i] == True:
                ax.scatter(l, x1y4[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x1y4[i][0], s=50, color='red', facecolors='none')
        if x2y1[i] != None:
            if stablex2y1[i] == True:
                ax.scatter(l, x2y1[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x2y1[i][0], s=50, color='red', facecolors='none')
        if x2y2[i] != None:
            if stablex2y2[i] == True:
                ax.scatter(l, x2y2[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x2y2[i][0], s=50, color='red', facecolors='none')
        if x2y3[i] != None:
            if stablex2y3[i] == True:
                ax.scatter(l, x2y3[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x2y3[i][0], s=50, color='red', facecolors='none')
        if x2y4[i] != None:
            if stablex2y4[i] == True:
                ax.scatter(l, x2y4[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x2y4[i][0], s=50, color='red', facecolors='none')
        if x3y1[i] != None:
            if stablex3y1[i] == True:
                ax.scatter(l, x3y1[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x3y1[i][0], s=50, color='red', facecolors='none')
        if x3y2[i] != None:
            if stablex3y2[i] == True:
                ax.scatter(l, x3y2[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x3y2[i][0], s=50, color='red', facecolors='none')

        if x3y3[i] != None:
            if stablex3y3[i] == True:
                ax.scatter(l, x3y3[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x3y3[i][0], s=50, color='red', facecolors='none')
        if x3y4[i] != None:
            if stablex3y4[i] == True:
                ax.scatter(l, x3y4[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x3y4[i][0], s=50, color='red', facecolors='none')
        if x4y1[i] != None:
            if stablex4y1[i] == True:
                ax.scatter(l, x4y1[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x4y1[i][0], s=50, color='red', facecolors='none')
        if x4y2[i] != None:
            if stablex4y2[i] == True:
                ax.scatter(l, x4y2[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x4y2[i][0], s=50, color='red', facecolors='none')
        if x4y3[i] != None:
            if stablex4y3[i] == True:
                ax.scatter(l, x4y3[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x4y3[i][0], s=50, color='red', facecolors='none')
        if x4y4[i] != None:
            if stablex4y4[i] == True:
                ax.scatter(l, x4y4[i][0], s=50, color='blue')
            else:
                ax.scatter(l, x4y4[i][0], s=50, color='red', facecolors='none')
        
    plt.grid()
    plt.show()


def plot_Y_eq():
    # Same as above, but for y-coordinates
    fig, ax = plt.subplots(figsize=(10, 10))
    # plt.rcParams.update({'dpi': 600})
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$y$')
    # ax.set_xlim(-2, 2)

    for i, l in enumerate(lambdas):
        if x1y1[i] != None:
            if stablex1y1[i] == True:
                ax.scatter(l, x1y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x1y1[i][1], s=50, color='red', facecolors='none')
        if x1y2[i] != None:
            if stablex1y2[i] == True:
                ax.scatter(l, x1y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x1y2[i][1], s=50, color='red', facecolors='none')
        if x1y3[i] != None:
            if stablex1y3[i] == True:
                ax.scatter(l, x1y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x1y3[i][1], s=50, color='red', facecolors='none')
        if x1y4[i] != None:
            if stablex1y4[i] == True:
                ax.scatter(l, x1y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x1y4[i][1], s=50, color='red', facecolors='none')
        if x2y1[i] != None:
            if stablex2y1[i] == True:
                ax.scatter(l, x2y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x2y1[i][1], s=50, color='red', facecolors='none')
        if x2y2[i] != None:
            if stablex2y2[i] == True:
                ax.scatter(l, x2y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x2y2[i][1], s=50, color='red', facecolors='none')
        if x2y3[i] != None:
            if stablex2y3[i] == True:
                ax.scatter(l, x2y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x2y3[i][1], s=50, color='red', facecolors='none')
        if x2y4[i] != None:
            if stablex2y4[i] == True:
                ax.scatter(l, x2y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x2y4[i][1], s=50, color='red', facecolors='none')
        if x3y1[i] != None:
            if stablex3y1[i] == True:
                ax.scatter(l, x3y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x3y1[i][1], s=50, color='red', facecolors='none')
        if x3y2[i] != None:
            if stablex3y2[i] == True:
                ax.scatter(l, x3y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x3y2[i][1], s=50, color='red', facecolors='none')

        if x3y3[i] != None:
            if stablex3y3[i] == True:
                ax.scatter(l, x3y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x3y3[i][1], s=50, color='red', facecolors='none')
        if x3y4[i] != None:
            if stablex3y4[i] == True:
                ax.scatter(l, x3y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x3y4[i][1], s=50, color='red', facecolors='none')
        if x4y1[i] != None:
            if stablex4y1[i] == True:
                ax.scatter(l, x4y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x4y1[i][1], s=50, color='red', facecolors='none')
        if x4y2[i] != None:
            if stablex4y2[i] == True:
                ax.scatter(l, x4y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x4y2[i][1], s=50, color='red', facecolors='none')
        if x4y3[i] != None:
            if stablex4y3[i] == True:
                ax.scatter(l, x4y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x4y3[i][1], s=50, color='red', facecolors='none')
        if x4y4[i] != None:
            if stablex4y4[i] == True:
                ax.scatter(l, x4y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, x4y4[i][1], s=50, color='red', facecolors='none')
        
    plt.grid()
    plt.show()



def flow_rate_plot():
    # For every equilibria, plot the flow rate 2-x-y as a function of lambda
    # Displaying stable equilibria as blue, filled dots and unstable equilibria as red, unfilled circles

    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$2-x-y$')
    ax.set_xlim(-2, 2)
    #ax.set_ylim(-0.6, 1.6)

    for i, l in enumerate(lambdas):
        if x1y1[i] != None:
            if stablex1y1[i] == True:
                ax.scatter(l, 2-x1y1[i][0]-x1y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x1y1[i][0]-x1y1[i][1], s=50, color='red', facecolors='none')
        if x1y2[i] != None:
            if stablex1y2[i] == True:
                ax.scatter(l, 2-x1y2[i][0]-x1y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x1y2[i][0]-x1y2[i][1], s=50, color='red', facecolors='none')
        if x1y3[i] != None:
            if stablex1y3[i] == True:
                ax.scatter(l, 2-x1y3[i][0]-x1y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x1y3[i][0]-x1y3[i][1], s=50, color='red', facecolors='none')
        if x1y4[i] != None:
            if stablex1y4[i] == True:
                ax.scatter(l, 2-x1y4[i][0]-x1y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x1y4[i][0]-x1y4[i][1], s=50, color='red', facecolors='none')
        if x2y1[i] != None:
            if stablex2y1[i] == True:
                ax.scatter(l, 2-x2y1[i][0]-x2y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x2y1[i][0]-x2y1[i][1], s=50, color='red', facecolors='none')
        if x2y2[i] != None:
            if stablex2y2[i] == True:
                ax.scatter(l, 2-x2y2[i][0]-x2y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x2y2[i][0]-x2y2[i][1], s=50, color='red', facecolors='none')
        if x2y3[i] != None:
            if stablex2y3[i] == True:
                ax.scatter(l, 2-x2y3[i][0]-x2y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x2y3[i][0]-x2y3[i][1], s=50, color='red', facecolors='none')
        if x2y4[i] != None:
            if stablex2y4[i] == True:
                ax.scatter(l, 2-x2y4[i][0]-x2y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x2y4[i][0]-x2y4[i][1], s=50, color='red', facecolors='none')
        if x3y1[i] != None:

            if stablex3y1[i] == True:
                ax.scatter(l, 2-x3y1[i][0]-x3y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x3y1[i][0]-x3y1[i][1], s=50, color='red', facecolors='none')
        if x3y2[i] != None:
            if stablex3y2[i] == True:
                ax.scatter(l, 2-x3y2[i][0]-x3y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x3y2[i][0]-x3y2[i][1], s=50, color='red', facecolors='none')
        
        if x3y3[i] != None:
            if stablex3y3[i] == True:
                ax.scatter(l, 2-x3y3[i][0]-x3y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x3y3[i][0]-x3y3[i][1], s=50, color='red', facecolors='none')
        if x3y4[i] != None:
            if stablex3y4[i] == True:
                ax.scatter(l, 2-x3y4[i][0]-x3y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x3y4[i][0]-x3y4[i][1], s=50, color='red', facecolors='none')
        if x4y1[i] != None:
            if stablex4y1[i] == True:
                ax.scatter(l, 2-x4y1[i][0]-x4y1[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x4y1[i][0]-x4y1[i][1], s=50, color='red', facecolors='none')
        if x4y2[i] != None:
            if stablex4y2[i] == True:
                ax.scatter(l, 2-x4y2[i][0]-x4y2[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x4y2[i][0]-x4y2[i][1], s=50, color='red', facecolors='none')
        if x4y3[i] != None:
            if stablex4y3[i] == True:
                ax.scatter(l, 2-x4y3[i][0]-x4y3[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x4y3[i][0]-x4y3[i][1], s=50, color='red', facecolors='none')
        if x4y4[i] != None:
            if stablex4y4[i] == True:
                ax.scatter(l, 2-x4y4[i][0]-x4y4[i][1], s=50, color='blue')
            else:
                ax.scatter(l, 2-x4y4[i][0]-x4y4[i][1], s=50, color='red', facecolors='none')
    
    plt.grid()
    plt.show()


     

def get_all_unstable():
    # Function that iterates through all lambda, goes through all equilibria points at that lambda, checks if they are stable or not, and returns a list of all unstable equilibria points
    # Returns a list of 20 lists of tuples. Each list of tuples corresponds to a lambda, and each tuple is an unstable equilibrium point at that lambda

    unstable = []

    for i, l in enumerate(lambdas):

        unstable_at_lambda = []

        if stablex1y1[i] == False:
            unstable_at_lambda.append(x1y1[i])
        if stablex1y2[i] == False:
            unstable_at_lambda.append(x1y2[i])
        if stablex1y3[i] == False:
            unstable_at_lambda.append(x1y3[i])
        if stablex1y4[i] == False:
            unstable_at_lambda.append(x1y4[i])
        if stablex2y1[i] == False:
            unstable_at_lambda.append(x2y1[i])
        if stablex2y2[i] == False:
            unstable_at_lambda.append(x2y2[i])
        if stablex2y3[i] == False:
            unstable_at_lambda.append(x2y3[i])
        if stablex2y4[i] == False:
            unstable_at_lambda.append(x2y4[i])
        if stablex3y1[i] == False:
            unstable_at_lambda.append(x3y1[i])
        if stablex3y2[i] == False:
            unstable_at_lambda.append(x3y2[i])
        if stablex3y3[i] == False:
            unstable_at_lambda.append(x3y3[i])
        if stablex3y4[i] == False:
            unstable_at_lambda.append(x3y4[i])
        if stablex4y1[i] == False:
            unstable_at_lambda.append(x4y1[i])
        if stablex4y2[i] == False:
            unstable_at_lambda.append(x4y2[i])
        if stablex4y3[i] == False:
            unstable_at_lambda.append(x4y3[i])
        if stablex4y4[i] == False:
            unstable_at_lambda.append(x4y4[i])
        
        unstable.append(unstable_at_lambda)
    
    return unstable

def get_all_stable():

    # Same as above, but for stable equilibria

    stable = []

    for i, l in enumerate(lambdas):
            
            stable_at_lambda = []
    
            if stablex1y1[i] == True:
                stable_at_lambda.append(x1y1[i])
            if stablex1y2[i] == True:
                stable_at_lambda.append(x1y2[i])
            if stablex1y3[i] == True:
                stable_at_lambda.append(x1y3[i])
            if stablex1y4[i] == True:
                stable_at_lambda.append(x1y4[i])
            if stablex2y1[i] == True:
                stable_at_lambda.append(x2y1[i])
            if stablex2y2[i] == True:
                stable_at_lambda.append(x2y2[i])
            if stablex2y3[i] == True:
                stable_at_lambda.append(x2y3[i])
            if stablex2y4[i] == True:
                stable_at_lambda.append(x2y4[i])
            if stablex3y1[i] == True:
                stable_at_lambda.append(x3y1[i])
            if stablex3y2[i] == True:
                stable_at_lambda.append(x3y2[i])
            if stablex3y3[i] == True:
                stable_at_lambda.append(x3y3[i])
            if stablex3y4[i] == True:
                stable_at_lambda.append(x3y4[i])
            if stablex4y1[i] == True:
                stable_at_lambda.append(x4y1[i])
            if stablex4y2[i] == True:
                stable_at_lambda.append(x4y2[i])
            if stablex4y3[i] == True:
                stable_at_lambda.append(x4y3[i])
            if stablex4y4[i] == True:
                stable_at_lambda.append(x4y4[i])
            
            stable.append(stable_at_lambda)

    return stable

def plot_unstable():
    # Function that plots all unstable equilibria as a function of lambda
    # Plots stable equilibria as blue, filled dots and unstable equilibria as red, unfilled circles

    unstable = get_all_unstable()
    
    fig, ax = plt.subplots(figsize=(10, 10))
    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$x$')
    ax.set_xlim(-2, 2)

    for i, l in enumerate(lambdas):
        if unstable[i][0] != None:
            for j in unstable[i]:
                ax.scatter(l, 2-j[0]-j[1], s=50, color='red', facecolors='none')
        
    plt.grid()
    plt.show()


def plot_flow_final():
    
    # Function that plots all unstable and stable flow rates (2-x-y).
    # dashed red lines between unstable points, solid blue lines between stable points, and solid black lines between stable and unstable points

    unstable = get_all_unstable()

    stable = get_all_stable()

    fig, ax = plt.subplots(figsize=(10, 10))

    ax.set_xlabel(r'$\lambda$')
    ax.set_ylabel(r'$2-x-y$')

    ax.set_xlim(-2, 2)

    # for i, l in enumerate(lambdas):
    #     if unstable[i][0] != None:
    #         for j in unstable[i]:
    #             ax.scatter(l, 2-j[0]-j[1], s=50, color='red', facecolors='none')
    #     if stable[i][0] != None:
    #         for j in stable[i]:
    #             ax.scatter(l, 2-j[0]-j[1], s=50, color='blue')


    for i, l in enumerate(lambdas): # For all lambda
        print(f"Key: {i}")
        print(f"lambda = {l}")
        for j, eqm in enumerate(all_equilibria): # For all equilibria, at this lambda
            if eqm[i] != None: # If there is a real equilibrium value at this lambda
                if all_stability[i][j]: # If this equilibrium point is stable
                    if j == 0 or i==0: # If this is the first equilibrium point, just make a scatter point
                        ax.scatter(l, 2-eqm[i][0]-eqm[i][1], s=50, color='blue') # Plot it as a blue dot
                    else: # If not the first equilibrium point
                        ax.plot([lambdas[i-1], l], [2-eqm[i-1][0]-eqm[i-1][1], 2-eqm[i][0]-eqm[i][1]], color='blue')
                        # Plot a blue line between this equilibrium point and the previous one
                else: # If this equilibrium point is unstable
                    if j == 0 or i==0:
                        ax.scatter(l, 2-eqm[i][0]-eqm[i][1], s=50, color='red', facecolors='none')
                    else:
                        # Checking if there is a previous equilibrium point
                        print([lambdas[i-1], l])
                        # print(all_equilibria[j-1][i-1][0])
                        
                        # print(2-eqm[i][0]-eqm[i][1])
                        print(eqm[i-1])
                        print(eqm[i])


                        ax.plot([lambdas[i-1], l], [2-eqm[i-1][0]-eqm[i-1][1], 2-eqm[i][0]-eqm[i][1]], color='red', linestyle='dashed')
                        # Plot a red dashed line between this equilibrium point and the previous one
                    



                

        
    plt.grid()
    plt.show()


# plot_flow_final()



# plt.rcParams['figure.figsize'] = [10, 10]
# plt.rcParams['figure.dpi'] = 600

# plot_equilibria()

plot_eigenvalues(eigenvalsx3y1)

# plot_eq_3D()

# plot_X_eq()
# plot_Y_eq()

# flow_rate_plot()

# flow_rate_plot2()