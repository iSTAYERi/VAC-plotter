import glob
import os


class XmlSearcher:

    def __init__(self, file_path, VAC_type_forward):
        self.file_path = file_path
        self.VAC_type_forward = VAC_type_forward
        self.list_of_files = []
        self.search_files()

    def search_files(self):
        os.chdir(self.file_path)

        if self.VAC_type_forward is True:
            self.list_of_files = glob.glob("*frw.xml")
        else:
            self.list_of_files = glob.glob("*bck.xml")

        self.list_of_files.sort()
