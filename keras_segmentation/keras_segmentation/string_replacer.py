# -*- coding: utf-8 -*-
"""
Created on Wed May  6 15:45:19 2020

@author: UC
"""

import os

path = 'C:/Users/UC/Downloads/sunrgb/seg'
i = 0
for filename in os.listdir(path):
    file = filename.replace('seg','img')
    os.rename(os.path.join(path,filename), os.path.join(path,file))