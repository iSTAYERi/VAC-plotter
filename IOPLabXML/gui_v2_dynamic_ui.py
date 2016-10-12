#!/usr/bin/python
# -*- coding: utf-8 -*-
# 123

from __future__ import unicode_literals
from PyQt5 import QtCore, QtWidgets, uic
import sys
import matplotlib
from matplotlib.backends.backend_qt5agg import (
    NavigationToolbar2QT as NavigationToolbar)

import XML_parser
import plotter
import xml_searcher


class MainWindow(QtWidgets.QMainWindow):

    def init_variables(self):

        self.vertical_layout_scroll = QtWidgets.QVBoxLayout(self)
        self.layout_plot = QtWidgets.QVBoxLayout(self)
        self.layout_plot_model = QtWidgets.QVBoxLayout(self)

        self.scroll_area = QtWidgets.QScrollArea(self)
        self.action_quit = QtWidgets.QAction(self)
        self.action_about = QtWidgets.QAction(self)

        self.btn_browse = QtWidgets.QPushButton(self)
        self.btn_plot = QtWidgets.QPushButton(self)
        self.btn_clean = QtWidgets.QPushButton(self)
        self.btn_accept_browse = QtWidgets.QPushButton(self)
        self.btn_plot_model = QtWidgets.QPushButton(self)
        self.btn_clean_model = QtWidgets.QPushButton(self)

        self.radio_btn_frw = QtWidgets.QRadioButton(self)
        self.radio_btn_bck = QtWidgets.QRadioButton(self)

        self.text_browse = QtWidgets.QPlainTextEdit(self)
        self.line_edit_file_path = QtWidgets.QLineEdit(self)

        self.progress_bar = QtWidgets.QProgressBar(self)
        self.progress_bar_model = QtWidgets.QProgressBar(self)

    def __init__(self):

        QtWidgets.QMainWindow.__init__(self)
        uic.loadUi('gui.ui', self)

        self.btn_browse.clicked.connect(self.file_browse)
        self.btn_plot.clicked.connect(self.parse_and_plot_button)
        self.btn_clean.clicked.connect(self.clean_plot_button)
        self.btn_accept_browse.clicked.connect(self.file_browse_accept)

        self.action_quit.setShortcut(QtCore.Qt.CTRL + QtCore.Qt.Key_Q)
        self.action_quit.triggered.connect(self.fileQuit)
        self.action_about.triggered.connect(self.about)

        self.sc1 = plotter.XMLDataMplCanvas(self, width=4, height=3, dpi=80)
        self.navi_toolbar = NavigationToolbar(self.sc1, self)
        self.layout_plot.addWidget(self.sc1)
        self.layout_plot.addWidget(self.navi_toolbar)

        self.sc2 = plotter.XMLDataMplCanvas(self, dpi=80)
        self.navi_toolbar2 = NavigationToolbar(self.sc2, self)
        self.layout_plot_model.addWidget(self.sc2)
        self.layout_plot_model.addWidget(self.navi_toolbar2)

    def file_browse(self):

        directory = QtWidgets.QFileDialog.\
            getExistingDirectory(parent=self,
                                 caption='Open catalog', directory='/home')
        self.line_edit_file_path.setText(directory)

    def file_browse_accept(self):
        self.file_path = self.line_edit_file_path.text()

        if self.radio_btn_frw.isChecked():
            VAC_type_forward = True
        else:
            VAC_type_forward = False

        self.xsearch = xml_searcher.XmlSearcher(self.file_path,
                                                VAC_type_forward)

        self.crean_layout(self.vertical_layout_scroll)
        self.add_checkboxes()

# Функция, удаляющая виджеты из layout'ов.
    def crean_layout(self, layout):
        for i in reversed(range(layout.count())):
            layout.itemAt(i).widget().setParent(None)

    def add_checkboxes(self):
        self.checks = []
        for i in self.xsearch.list_of_files:
            c = QtWidgets.QCheckBox("%s" % i)
            c.setCheckState(True)
            c.setTristate(False)
            self.vertical_layout_scroll.addWidget(c)
            self.checks.append(c)

        print(self.checks)
        self.scroll_area.setWidgetResizable(True)

    def fileQuit(self):
        self.close()

    def closeEvent(self, ce):
        self.fileQuit()

    def about(self):
        QtWidgets.QMessageBox.about(self, "Справка",
                                    "...---...ФЦП №117...---...")

    def parse_and_plot_button(self):
        u_str = ''
        i_str = ''

        self.check_list_of_files(self.xsearch.list_of_files)

        self.progress_bar.setMinimum(0)
        self.progress_bar.setMaximum(len(self.filtered_list) - 1)

        for idx, list_of_files in enumerate(self.filtered_list):
            file_name = self.file_path + '/' + list_of_files
            ltex = XML_parser.XMLPloter(file_name)

            if self.radio_btn_frw.isChecked():
                u_load = list(map(lambda x: float(x)*1e3, ltex.u_load))
                i_load = list(map(lambda x: float(x)*1e6, ltex.i_load))
                u_str = 'U, mV'
                i_str = 'I, uA'
            else:
                u_load = ltex.u_load
                i_load = list(map(lambda x: float(x)*1e6, ltex.i_load))
                u_str = 'U, V'
                i_str = 'I, uA'
            self.sc1.update_figure(u_load, u_str, i_load, i_str)
            self.progress_bar.setValue(idx)

    def check_list_of_files(self, files_list=[]):
        self.filtered_list = []
        for idx, x in enumerate(self.checks):
            if x.isChecked():
                self.filtered_list.append(files_list[idx])
#        print("Enabled files: ", self.filtered_list)

    def clean_plot_button(self):
        self.sc1.clean_figure()
        self.progress_bar.setValue(0)


if __name__ == "__main__":

    matplotlib.use('Qt5Agg')
    app = QtWidgets.QApplication(sys.argv)
    mw = MainWindow()
    mw.show()
    sys.exit(app.exec_())