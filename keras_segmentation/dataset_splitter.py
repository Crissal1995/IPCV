# -*- coding: utf-8 -*-
"""
Created on Wed May  6 16:09:08 2020

@author: UC
"""


import os
import numpy as np
import shutil

# # Creating Train / Val / Test folders (One time use)
root_dir = 'C:/Users/UC/Downloads/sunrgb'
rgbFolder = '/rgb'
segFolder = '/seg'
matFolder = '/mat'

os.makedirs(root_dir +'/train' + rgbFolder)
os.makedirs(root_dir +'/train' + segFolder)
os.makedirs(root_dir +'/train' + matFolder)
os.makedirs(root_dir +'/val' + rgbFolder)
os.makedirs(root_dir +'/val' + segFolder)
os.makedirs(root_dir +'/val' + matFolder)
os.makedirs(root_dir +'/test' + rgbFolder)
os.makedirs(root_dir +'/test' + segFolder)
os.makedirs(root_dir +'/test' + matFolder)

# Creating partitions of the data after shuffeling
currentCls = rgbFolder # TO CHANGE
src = root_dir+currentCls # Folder to copy images from
allFileNames1 = os.listdir(src)

currentCls = segFolder
src = root_dir+currentCls # Folder to copy images from
allFileNames2 = os.listdir(src)

currentCls = matFolder
src = root_dir+currentCls # Folder to copy images from
allFileNames3 = os.listdir(src)

allFileNames = list(zip(allFileNames1,allFileNames2,allFileNames3))


np.random.shuffle(allFileNames)
allFileNames1, allFileNames2, allFileNames3 = zip(*allFileNames)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames1),
                                                          [int(len(allFileNames1)*0.7), int(len(allFileNames1)*0.85)])

#allFileNames1, allFileNames2, allFileNames3 = zip(*allFileNames)
#print(train_FileNames.shape)
currentCls = rgbFolder
src = root_dir+currentCls # Folder to copy images from
train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]

print('Total images: ', len(allFileNames1))
print('Training: ', len(train_FileNames))
print('Validation: ', len(val_FileNames))
print('Testing: ', len(test_FileNames))


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, root_dir +'/train'+currentCls)

for name in val_FileNames:
    shutil.copy(name, root_dir +'/val'+currentCls)

for name in test_FileNames:
    shutil.copy(name, root_dir +'/test'+currentCls)
    



#np.random.shuffle(allFileNames2)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames2),
                                                          [int(len(allFileNames2)*0.7), int(len(allFileNames2)*0.85)])
currentCls = segFolder
src = root_dir+currentCls
train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, root_dir +'/train'+currentCls)

for name in val_FileNames:
    shutil.copy(name, root_dir +'/val'+currentCls)

for name in test_FileNames:
    shutil.copy(name, root_dir +'/test'+currentCls)
    

#np.random.shuffle(allFileNames3)
train_FileNames, val_FileNames, test_FileNames = np.split(np.array(allFileNames3),
                                                          [int(len(allFileNames3)*0.7), int(len(allFileNames3)*0.85)])
currentCls = matFolder
src = root_dir+currentCls
train_FileNames = [src+'/'+ name for name in train_FileNames.tolist()]
val_FileNames = [src+'/' + name for name in val_FileNames.tolist()]
test_FileNames = [src+'/' + name for name in test_FileNames.tolist()]


# Copy-pasting images
for name in train_FileNames:
    shutil.copy(name, root_dir +'/train'+currentCls)

for name in val_FileNames:
    shutil.copy(name, root_dir +'/val'+currentCls)

for name in test_FileNames:
    shutil.copy(name, root_dir +'/test'+currentCls)