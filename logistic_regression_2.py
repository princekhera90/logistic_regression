# -*- coding: utf-8 -*-
"""
Created on Wed Jul 18 15:50:17 2018

@author: prince khera
"""
# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:01:54 2018

@author: prince khera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logistic(x,t):
    y = t[0,0] + t[0,1]*(x[1]**2) + t[0,2]*(x[2]**2) + t[0,3]*x[1] + t[0,4]*x[2]
    h = 1/(1+np.exp(-y))
    return h

def cost(x,y,t):
    h = logistic(x,t)
    j = -(np.log(h).dot(y.T) + np.log(1-h).dot((1-y).T))/m
    return j[0]

def gradient(x,y,t,ne,a):
    jh = []
    for i in range(ne):
        d = logistic(x,t) - y
        t[0,0] = t[0,0] - (a*np.sum(d)/max(y.shape))
        t[0,1] = t[0,1] - a*(d.dot(x[1]**2)/max(y.shape))
        t[0,2] = t[0,2] - a*(d.dot(x[2]**2)/max(y.shape))
        t[0,3] = t[0,3] - a*(d.dot(x[1])/max(y.shape))
        t[0,4] = t[0,4] - a*(d.dot(x[2])/max(y.shape))
        jh.append(cost(x,y,t))
    return jh,t

df = pd.read_csv('ex2data2.txt',names=['a','b','c'])

m = len(df)
n = len(df.columns) - 1
x = np.ones(m)

X = np.array([x,df['a'].values,df['b'].values])
Y = np.array([df['c'].values])
t = np.zeros((1,5))
h = logistic(X,t)
j = cost(X,Y,t)
jh,T = gradient(X,Y,t,100000,.001)
y_p = logistic(X,T)
#plt.scatter(T.dot(X),y_p)
for i in range(max(Y.shape)):
    if Y[0,i]==0:
        plt.scatter(X[1,i],X[2,i],marker = '+')
    else:
        plt.scatter(X[1,i],X[2,i],marker = '_')
        

#plt.plot(X[1],((-T[0,0] - T[0,1]*(X[1]**2) - T[0,3]*X[1])/(T[0,4] + T[0,2]*(X[2]))))


xb1 = np.linspace(-1.0, 1.0, 100)
xb2 = np.linspace(-1.0, 1.0, 100)
Xb1, Xb2 = np.meshgrid(xb1,xb2)
b = T[0,1]*(Xb1**2) + T[0,2]*(Xb2**2) + T[0,3]*Xb1 + T[0,4]*Xb2 + T[0,0]
plt.contour(Xb1,Xb2,b,[0], colors='r')


