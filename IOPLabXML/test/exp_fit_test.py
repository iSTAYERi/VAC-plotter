#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sat Oct 15 13:11:05 2016

@author: stayer
"""

from __future__ import division

import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as plt


def func(x, a, b, c):
    return a*np.exp(-b*x)-c


xData = np.load('xData.npy')*10**5
yData = np.load('yData.npy')*10**5

print(xData.min(), xData.max())
print(yData.min(), yData.max())

trialX = np.linspace(xData[0], xData[-1], 1000)

# Fit a polynomial
fitted = np.polyfit(xData, yData, 10)[::-1]
y = np.zeros(len(trialX))
for i in range(len(fitted)):
    y += fitted[i]*trialX**i

# Fit an exponential
popt, pcov = optimize.curve_fit(func, xData, yData)
print(popt)
yEXP = func(trialX, *popt)

plt.figure()
plt.plot(xData, yData, label='Data', marker='o')
plt.plot(trialX, yEXP, 'r-', ls='--', label="Exp Fit")
plt.plot(trialX, y, label='10 Deg Poly')
plt.legend()
plt.show()
