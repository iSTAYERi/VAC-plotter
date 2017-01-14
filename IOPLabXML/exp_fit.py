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

        ExpPlot.print_coeffs(float(popt[0]), float(popt[1]))

        return float(popt[0])

    def print_coeffs(a, b):

        # print("коэффициент А = ")
        print(a)

        # print("коэффициент B = ")
        # print(b)
