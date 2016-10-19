# -*- coding: utf-8 -*-

from scipy.optimize import curve_fit
import numpy as np


class ExpPlot:

    def _exp_fun(x, a, b):
        return a * (np.exp(b*x)-1)

    def _get_exp_fit(U, I):
        guess = [1, 0.03]
        params, cov = curve_fit(ExpPlot._exp_fun, U, I, p0=guess)
        return params

    def fit_and_get_coeff_a(xData, yData):
        xData = np.array(xData)
        yData = np.array(yData)

        popt = ExpPlot._get_exp_fit(xData, yData)
        return float(popt[1])
