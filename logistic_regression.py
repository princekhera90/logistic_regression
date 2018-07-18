# -*- coding: utf-8 -*-
"""
Created on Tue Jul 17 19:01:54 2018

@author: prince khera
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

def logistic(x,t):
    y = t.dot(x)
    h = 1/(1+np.exp(-y))
    return h

def cost(x,y,t):
    h = logistic(x,t)
    j = -(np.log(h).dot(y.T) + np.log(1-h).dot((1-y).T))/m
    return j[0,0]

def gradient(x,y,t,ne,a):
    jh = []
    for i in range(ne):
        d = logistic(x,t) - y
        t = t - (a*(x.dot(d.T))/max(y.shape)).T
        jh.append(cost(x,y,t))
    return jh,t

df = pd.read_csv('ex2data1.txt',names=['a','b','c'])

m = len(df)
n = len(df.columns) - 1
x = np.ones(m)

X = np.array([x,df['a'].values,df['b'].values])
Y = np.array([df['c'].values])
t = np.zeros((1,n+1))
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
        

plt.plot(X[1],((-T[0,0] - T[0,1]*X[1])/T[0,2]))

