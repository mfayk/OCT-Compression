#!/usr/bin/env python

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import os
import numpy as np
from ipywidgets import widgets,interact,IntProgress
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
#from OctCorrection import *
#from ImageProcessing import *

from mpl_toolkits.mplot3d import Axes3D
import os
import glob
import re
import pickle
from PIL import Image
import pandas as pd
#import cv2
from tifffile import imsave, imread
from numba import jit

import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import rawpy
import imageio


from oct_converter.readers import FDS
from struct import unpack

from skimage.metrics import structural_similarity

import time

import cv2


mode = 'Ldiff'

paths = ['10.20.22/ATLC3_Trial1/*.tif', '11.17.22/ATLC8b/*.tif', '3.10.23/ATLC3/*.tif', '3.10.23/HB/*.tif', '3.15.23/HB2.3.15.23/*.tif', '3.29.23/ATLC3/Static/*.tif', '3.29.23/ATLC3/Transfer/*.tif', '3.29.23/HB2/Static/*.tif', '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

#srcpath='/scratch1/mfaykus/automated-oct-scans-acquisition/sample_scans/';
srcpath='/scratch/mfaykus/BioFilm/'

i=0

l=glob.glob(srcpath+paths[i])
l.sort()

print(paths[i])


input_data = np.array([np.array(Image.open(fname)) for fname in l])
input_data = input_data.astype(float)
data = input_data.copy()

print(np.max(input_data))
print(np.min(input_data))

i_data = input_data.copy()
print(i_data)

data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX)


print(data)

print(np.max(data))
print(np.min(data))

                
input_data = data.copy()
diff_data = input_data.copy()
decomp_data = input_data.copy()
de_data = input_data.copy()


d=1
if(mode == "Ldiff"):
    while d < len(l):
        diff_data[d] =  input_data[d] - input_data[d-1]
        d=d+1
elif(mode == "0diff"):
    while d < len(l):
        diff_data[d] =  input_data[d] - input_data[0]
        d=d+1
input_data = diff_data

#################################################################
d=1
if(mode == "Ldiff"):
    while d < len(l):
        de_data[d] =  decomp_data[d] + decomp_data[d-1]
        d=d+1
        
elif(mode == "0diff"):
    while d < len(l):
        de_data[d] =  decomp_data[d] + decomp_data[0]
        d=d+1

        
decomp_data = de_data      
        
new_min = 0
new_max = 255

D_data = decomp_data.copy()
D_data = cv2.normalize(decomp_data, None, 0, 255, cv2.NORM_MINMAX)

print(np.max(D_data))
print(np.min(D_data))    
    
D_data = D_data.astype(np.uint8)

print(D_data)

print(np.max(D_data))
print(np.min(D_data))

num=0
for img in D_data:
    path = 'tiff_data/'
    im = Image.fromarray(img)
    im.save(path + str(num)  + '.tif')
    #imageio.imwrite(path + str(num)  + '.tif', img)
    num +=1