# -*- coding: utf-8 -*-
"""
Created on Tue May 12 16:06:01 2020

@author: UC
"""


from PIL import Image 
import glob, os

def convert(directory):
    for infile in glob.glob(directory+"*.JPG"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(file + ".png", "PNG")
    for infile in glob.glob(directory+"*.jpg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(file + ".png", "PNG") 
    for infile in glob.glob(directory+"*.JPEG"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(directory + file + ".png", "PNG")
    for infile in glob.glob(directory+"*.jpeg"):
        file, ext = os.path.splitext(infile)
        im = Image.open(infile)
        rgb_im = im.convert('RGB')
        rgb_im.save(file + ".png", "PNG")
        
        
directory = "C:/Users/UC/Desktop/image-segmentation-keras-master/sunrgb/train/rgb/"
convert(directory)
directory = "C:/Users/UC/Desktop/image-segmentation-keras-master/sunrgb/test/rgb/"
convert(directory)
directory = "C:/Users/UC/Desktop/image-segmentation-keras-master/sunrgb/val/rgb/"
convert(directory)

