import numpy as np
import xlwings as xw

class ExcelDataManager:
    def __init__(self, book_name = "./DecisionTree/watermelon3.0.xlsx", sheet_name= "V1"):
        self.__wb = xw.Book(book_name)
        self.__sheet_name = sheet_name

    def get_column(self, column_name):
        wb = self.__wb
        sht = wb.sheets[self.__sheet_name]

        title_list: list = sht.range(f"A1:Z1").value
        index_name = str(chr(title_list.index(column_name) + 65))
        density_list: list = sht.range(f"{index_name}2:{index_name}65536").value
        return density_list[:density_list.index(None)]

    def get_all_title(self):
        wb = self.__wb
        sht = wb.sheets[self.__sheet_name]

        title_list: list = sht.range(f"A1:ZZ1").value
        width = title_list.index(None)
        return title_list[:width]

    def fetch(self)->list:
        wb = self.__wb
        sht = wb.sheets[self.__sheet_name]

        title_list: list = sht.range(f"A1:ZZ1").value
        width = title_list.index(None)

        density_list: list = sht.range(f"A1:A65536").value
        height = density_list.index(None)

        return sht.range(f"A2:{str(chr(width + 65 - 1))}{height}").value