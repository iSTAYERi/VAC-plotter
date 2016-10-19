
from PyQt5 import QtWidgets
from numpy import arange, sin, pi
from matplotlib.figure import Figure
from matplotlib.backends.backend_qt5agg import (
    FigureCanvasQTAgg as FigureCanvas,
    NavigationToolbar2QT as NavigationToolbar)
import numpy as np


class MyMplCanvas(FigureCanvas):

    def __init__(self, parent=None, width=5, height=4, dpi=100):
        fig = Figure(figsize=(width, height), dpi=dpi)
        self.axes = fig.add_subplot(111)
        self.axes.hold(True)

        self.compute_initial_figure()

        FigureCanvas.__init__(self, fig)
        self.setParent(parent)

        FigureCanvas.setSizePolicy(self,
                                   QtWidgets.QSizePolicy.Expanding,
                                   QtWidgets.QSizePolicy.Expanding)
        FigureCanvas.updateGeometry(self)

    def compute_initial_figure(self):
        pass

    def update_figure(self, u_load, u_str, i_load, i_str):
        pass


class XMLDataMplCanvas(MyMplCanvas):

    def compute_initial_figure(self):

        self.axes.plot(0, 0)
        self.axes.set_xlabel("U, V")
        self.axes.set_ylabel("I, A")
#         self.axes.grid()

    def clean_figure(self):

        self.axes.hold(False)
        self.compute_initial_figure()
        self.draw()

    def clean_model(self):

        self.axes.hold(False)
        self.axes.bar([0, 1, 2, 3, 4, 5, 6, 7, 8, 9],
                      [1, 1, 1, 1, 1, 1, 1, 1, 1, 1])
        self.axes.set_xlabel("model")
        self.axes.set_ylabel("a")
        self.draw()

    def update_figure(self, u_load, u_str, i_load, i_str, file_name):

        self.u_load = u_load
        self.i_load = i_load

        print(self.u_load)
        print(self.i_load)

        self.axes.hold(True)
        self.axes.plot(self.u_load, self.i_load, label=file_name)
        self.axes.set_xlabel(u_str)
        self.axes.set_ylabel(i_str)
        self.axes.grid(True)
        self.draw()

    def plot_stat_graph(self, a):

        b = [n + 1 for n in len(a)]

        self.axes.bar(b, self.a)
        self.axes.set_xlabel("model")
        self.axes.set_ylabel("a")
        self.draw()
