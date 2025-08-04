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

#from OCT_reader import get_OCTSpectralRawFrame

from oct_converter.readers import FDS


from struct import unpack


srcpath='/scratch/mfaykus/BioFilm/'
pathexp='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/output/';
if not os.path.exists(pathexp):
     os.makedirs(pathexp)
l=glob.glob(srcpath+'*/*.raw')
l.sort()
print(l)
print(len(l))

input_path=l[0]
print(input_path)

size = (250, 2048, 1000)
#size = (int(900/4),int(900/4),int(1024/4))
input_data=np.fromfile(input_path,dtype=np.uint8).reshape(size)  #load it 
input_data=np.flip(input_data, axis=2)
#input_data = np.ascontiguousarray(input_data)

diff_data = input_data.copy()
#input_data = input_data.astype(int)
#diff_data = diff_data.astype(int)

print("shape")
print(diff_data.shape)


print(input_data[0])
print(input_data[0].shape)

i=1
j=0
k=0
while i < 250:
    diff_data[i] = input_data[i-1] - input_data[i]
    i=i+1


#fig = plt.imshow(diff_data[1], interpolation='nearest')

print("base 0")
print(input_data[0])

print("sub")
print(input_data[1])

print("diff")
print(diff_data[1])