# -*- coding: utf-8 -*-
"""
Created on Wed Mar 10 22:04:00 2021

@author: Scott
"""
# https://stackoverflow.com/questions/60067953/is-it-possible-to-specify-the-pickle-protocol-when-writing-pandas-to-hdf5

import pickle

pickle.HIGHEST_PROTOCOL = 2
