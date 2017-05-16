#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Tue May 16 20:56:58 2017

@author: john
"""

import theano
import numpy
import theano.tensor as T
import matplotlib.pyplot as plt
rng = numpy.random

#Training Data
X = numpy.asarray([3,4,5,6.1,6.3,2.88,8.89,5.62,6.9,1.97,8.22,9.81,4.83,7.27,5.14,3.08])
Y = numpy.asarray([0.9,1.6,1.9,2.9,1.54,1.43,3.06,2.36,2.3,1.11,2.57,3.15,1.5,2.64,2.20,1.21])

X = X.astype(numpy.float32)
Y = Y.astype(numpy.float32)

m_value = rng.randn()
c_value = rng.randn()
#m = theano.shared(m_value,name ='ma')
#c = theano.shared(c_value,name ='ca')
m = theano.shared(m_value)
c = theano.shared(c_value)

#x = T.vector('xa')
#y = T.vector('ya')
x = T.vector()
y = T.vector()

num_samples = X.shape[0]

prediction = T.dot(x,m)+c
cost = T.sum(T.pow(prediction-y,2))/(2*num_samples)

gradm = T.grad(cost,m)
gradc = T.grad(cost,c)

learning_rate = 0.01
training_steps = 10000

train = theano.function([x,y],cost,updates = [(m,m-learning_rate*gradm),(c,c-learning_rate*gradc)])
test = theano.function([x],prediction)

for i in range(training_steps):
    costM = train(X,Y)
    print(costM)


print("Slope :")
print(m.get_value())
print("Intercept :")
print(c.get_value())

a = linspace(0,10,10)
b = test(a)
plt.plot(X,Y,'ropplt.plot(a,b)
