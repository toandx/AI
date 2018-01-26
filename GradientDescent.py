from __future__ import division, print_function, unicode_literals
import math
import numpy as np 
import matplotlib.pyplot as plt
def grad(x):
    return 2*x+ 5*np.cos(x)
def cost(x):
    return x**2 + 5*np.sin(x)
def myGD1(eta, x0):
    last=x0;
    for it in range(100):
        x_new = last - eta*grad(last)
        if abs(grad(x_new)) < 1e-3:
            break
        last=x_new;
    return (last, it)
(x1, it1) = myGD1(.1, -5)
(x2, it2) = myGD1(.1, 5)
print('Solution x1 = %f, cost = %f, obtained after %d iterations'%(x1, cost(x1), it1))
print('Solution x2 = %f, cost = %f, obtained after %d iterations'%(x2, cost(x2), it2))