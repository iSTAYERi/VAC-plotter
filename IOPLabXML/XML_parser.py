#!/usr//bin/python3
# -*- coding: utf-8 -*-

import lxml.etree as etree
import matplotlib.pyplot as plt
import codecs


class XMLPloter:

    def __init__(self, source_file_name):
        self.source_file_name = source_file_name
        self.target_file_name = '1.xml'
        self.u_load = []
        self.i_load = []

        self.encode_file()

        self.parse_and_get_values_out(path_out=self.target_file_name)

    def encode_file(self):
        BLOCKSIZE = 1048576  # or some other, desired size in bytes
        with codecs.open(self.source_file_name,
                         "r", "iso-8859-1") as sourceFile:
            with codecs.open(self.target_file_name,
                             "w", "utf-8") as targetFile:
                while True:
                    contents = sourceFile.read(BLOCKSIZE)
                    if not contents:
                        break
                    targetFile.write(contents)

    def scientificate_value(self, val):
        val = float(val)
        val = '%.3e' % val
        result = str(val)
        return result

    def xmlns_fol(self, folder=None, list_number=None):

        if (folder is not None) & (list_number is not None):
            xmlns_folder = "/*[name() = '" + folder + "']" + \
                "[" + str(list_number) + "]"

        elif folder is not None:
            xmlns_folder = "/*[name() = '" + folder + "']"

        return xmlns_folder

    def parse_and_get_values_out(self, path_out):

        tree = etree.parse(path_out)

        nodes_V = tree.xpath(self.xmlns_fol('LVData') +
                             self.xmlns_fol('Cluster') +
                             self.xmlns_fol('Cluster', 2) +
                             self.xmlns_fol('Array', 1) +
                             self.xmlns_fol('DBL') +
                             self.xmlns_fol('Val'))

        nodes_I = tree.xpath(self.xmlns_fol('LVData') +
                             self.xmlns_fol('Cluster') +
                             self.xmlns_fol('Cluster', 2) +
                             self.xmlns_fol('Array', 2) +
                             self.xmlns_fol('DBL') +
                             self.xmlns_fol('Val'))

        print("Количество нодов 'V': " + str(len(nodes_V)))
        print("Количество нодов 'I': " + str(len(nodes_I)))

        print("------------------")

        for x in nodes_V:
            self.u_load.append(x.text)
        for x in nodes_I:
            self.i_load.append(x.text)

    def plot_VAC(self):

        plt.plot(ltex.u_load, ltex.i_load)
        plt.ylabel("I")
        plt.xlabel("U")
        plt.show()

if __name__ == '__main__':

    # while(1):
        ltex = XMLPloter()

        sourceFileName = "xml/c1001_frw.xml"
        targetFileName = "xml/c1001_frw1.xml"

        BLOCKSIZE = 1048576  # or some other, desired size in bytes
        with codecs.open(sourceFileName, "r", "iso-8859-1") as sourceFile:
            with codecs.open(targetFileName, "w", "utf-8") as targetFile:
                while True:
                    contents = sourceFile.read(BLOCKSIZE)
                    if not contents:
                        break
                    targetFile.write(contents)

        ltex.parse_and_get_values_out(path_out="xml/c1001_frw1.xml")
        ltex.plot_VAC()

        print("Parse XML complete!")
