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
from matplotlib_scalebar.scalebar import ScaleBar
import seaborn as sns
import pandas as pd
import rawpy
import imageio
import bitstring

#from OCT_reader import get_OCTSpectralRawFrame

from oct_converter.readers import FDS


from struct import unpack


srcpath='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/';
srcpath='/scratch/mfaykus/BioFilm/'
pathexp='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/output/';
if not os.path.exists(pathexp):
     os.makedirs(pathexp)
l=glob.glob(srcpath+'*/*.raw')
l.sort()
#print(l)



#input_path = Path(__file__).parent / "/scratch/mfaykus/BioFilm/06.21.22/20220621_095224161_06.21.22 Control Processed Stack_processed.raw"

input_path="202_zfp_input_data.raw"

#size = (int(900/4),int(900/4),int(1024/4))
size = (250, 2048, 1000)
input_data=np.fromfile(input_path,dtype=np.uint8).reshape(size)  #load it 
#input_data=np.flip(input_data, axis=2)
#input_data = np.fromfile(input_path, dtype=np.uint8)

#input_data = np.ascontiguousarray(input_data)
#input_data = input_data.astype(float)

#print(type(input_data))
#print(input_data.dtype)
#input_data = input_data.reshape(100,500,500)


input_path=l[6]
print(input_path)
#exit(0)


#size = (int(900/4),int(900/4),int(1024/4))
#size = (2048, 1000, 250)
voxel_size=(10/size[0],10/size[1],2.23/size[2])



decomp_data=np.fromfile(input_path, dtype=np.uint8).reshape(size)  #load it 

    
#decomp_data=np.flip(decomp_data, axis=2)

#decomp_data = input_data.copy()




crop=int(500/4)  #find here the best position to crop the image and avoid spurious signals such as the Zspacer
bckHeight=int(100/4) #find here the position of the line above which the background intensity is calculated

fig, ax = plt.subplots(2,figsize=(20,10))

sl=1   #select the slice to be displayed (first)
im=input_data[sl,:,:]
ax[0].imshow(im.swapaxes(1,0), cmap='gray')
#ax[0].set_aspect(voxel_size[2]/voxel_size[0]) #set here the aspect ratio, 
                # you can calculate it from the z spacing and x,y spacing parameters in the .srm files
#ax[0].axhline(y=bckHeight, color='r', linestyle='-', label='background height')
#ax[0].axhline(y=crop, color='g', linestyle='-', linewidth=3,label='crop')
scalebar = ScaleBar(voxel_size[0]/1000,'mm') # set here the scale 1 pixel = 11 micron
ax[0].axis('off')
plt.gca().add_artist(scalebar)
ax[0].set_title('input');

sl=-1
im=decomp_data[sl,:,:]
ax[1].imshow(im.swapaxes(1,0), cmap='gray')
#ax[1].set_aspect(voxel_size[2]/voxel_size[0]) #set here the aspect ratio, 
                # you can calculate it from the z spacing and x,y spacing parameters in the .srm files
#ax[1].axhline(y=bckHeight, color='r', linestyle='-', label='background height')
#ax[1].axhline(y=crop, color='g', linestyle='-', linewidth=3,label='crop')
ax[1].legend(loc=2)
scalebar = ScaleBar(voxel_size[0]/1000,'mm') # set here the scale 1 pixel = 11 micron
ax[1].axis('off')
plt.gca().add_artist(scalebar)
ax[1].set_title('Decomp');
fig.savefig('Ascan.png') #save figure
