from lib2 import *

def initial_data_sum():
    dict_var = initial_data()
    a = 0
    for key in dict_var:
        a += dict_var[key]
    return a
