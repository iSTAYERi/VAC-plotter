# -*- coding: utf-8 -*-

from scipy.optimize import curve_fit
import numpy as np
import matplotlib.pyplot as plt


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
        return popt[0]


if __name__ == "__main__":

    xData = [0, 8.16318378415997,
             16.3264275145331, 24.4897616474028, 32.6530279784001,
             40.8162491081491, 48.9795832410188, 57.1428269713919,
             65.306059401453, 73.4693370327624, 81.63261466407171,
             89.7958922953811, 97.9591812270025, 106.12241365706001,
             114.28567998806, 122.44891241812, 130.61221265006,
             138.77550158168, 146.93871141111, 155.10195514149,
             163.26527797404, 171.42856690566998, 179.59184453697998,
             187.75505436640998, 195.91828679647, 204.08159832872,
             212.24486465972, 220.4081196904, 228.57132951984,
             236.73464105208, 244.89791868339, 253.06119631469997,
             261.22448524632, 269.3877289767, 277.55098400738,
             285.71425033838, 293.87750536907, 302.040805601]
    yData = [-0.000434385204429561, -0.000308482175079974,
             -0.00012765332216948998,
             0.000105342522981434, 0.000427874984622029, 0.0008645227417886511,
             0.00145742067282936, 0.00226049138991582, 0.00335875705828608,
             0.00483473181470513, 0.00684993269284055, 0.009591458616785599,
             0.0132955658687062, 0.0183251661686599, 0.025132707853193798,
             0.0343314038839807, 0.0467421319802662, 0.0634466048224979,
             0.08586060448374061, 0.11582402065186, 0.155696657317195,
             0.208471094053849, 0.277879026426946, 0.36846098918418496,
             0.48566316615490196, 0.6358539432379909, 0.8263171367250279,
             1.06519405784254, 1.36141780240601, 1.72459888472152,
             2.1650357713053,
             2.69361029609643, 3.32194792347628, 4.062435304467781,
             4.92843872601548, 5.93455967908692, 7.09682948168213,
             8.43316774643858]
    xData = list(map(lambda x: float("{:.3f}".format(x)), xData))
    yData = list(map(lambda x: float("{:.3f}".format(x)), yData))
    xData = np.array(xData)
    yData = np.array(yData)

    popt = ExpPlot.getExpFit(xData, yData)
#    poptR = ExpPlot.getExpFitR(xData, yData)
#    print(popt)

    trialX = np.linspace(xData[0], xData[-1], 1000)
    Iv = ExpPlot.expFun(trialX, *popt)
#    IvR = ExpPlot.expFunR(trialX, *poptR)

#    plt.figure()
#    plt.plot(xData, yData, 'b*')
    plt. figure()
    plt.plot(trialX, Iv, 'r-')
    plt.plot(xData, yData, 'b*')
#    plt.plot(trialX, IvR, 'g--')
#    plt. figure()
#    plt.plot(trialX, Iv, 'r-')
    plt.show()
