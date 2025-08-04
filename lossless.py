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



#srcpath='/scratch1/mfaykus/automated-oct-scans-acquisition/sample_scans/';
srcpath='/scratch/mfaykus/BioFilm/'
pathexp='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/output/';
if not os.path.exists(pathexp):
     os.makedirs(pathexp)
l=glob.glob(srcpath+'*/*.raw')
l.sort()
print(l)
print(len(l))

compressors = ['zstd','blosclz','lz4','lz4hc','zlib']
#compressors = ['zstd']

df = pd.DataFrame({
            'filename':[],
            'compressor':[],
            'CL':[],
            'cBW':[],
            'dBW':[],
            'CR':[]})
       
i = 0
while i < len(l):
    input_path=l[i]
    print(input_path)

    size = (250, 2048, 1000)
    #size = (int(900/4),int(900/4),int(1024/4))
    input_data=np.fromfile(input_path,dtype=np.uint8).reshape(size)  #load it 
    #print(type(input_data))
    input_data=np.flip(input_data, axis=2)
    #input_data = input_data.astype('int32')
    input_data = np.ascontiguousarray(input_data)
    diff_data = input_data.copy()
    decomp_data = input_data.copy()

    
    d=1
    while d < 250:
        diff_data[d] =  abs(input_data[d] - input_data[d-1])
        d=d+1
    input_data = diff_data
    
    
    for comp in compressors:
        j = 0
        while j < 9:
            compressor = libpressio.PressioCompressor.from_config({
                  # configure which compressor to use
            "compressor_id": "blosc",
                  # configure the set of metrics to be gathered
                  "early_config": {
                    "blosc:compressor": comp,
                    "blosc:metric": "composite",
                    "composite:plugins": ["time", "size", "error_stat", "external"]
                    },
                    "compressor_config": {
                            "blosc:clevel": j,

                }})
            
            comp_data = compressor.encode(input_data)
            decomp_data = compressor.decode(comp_data, decomp_data)
            metrics = compressor.get_metrics()
            #print(metrics)
            
            df.loc[len(df)] = [l[i], comp, j , (metrics['size:uncompressed_size']/1048576)/metrics['time:compress'], (metrics['size:uncompressed_size']/1048576)/metrics['time:decompress'], metrics['size:compression_ratio']]
            
            save = 0
            if save == 1:
                save_out = np.ndarray.tofile(comp_data,'202_zfp_comp_data.raw')
                save_out2 = np.ndarray.tofile(decomp_data,'202_zfp_decomp_data.raw')
                save_out3 = np.ndarray.tofile(input_data,'202_zfp_input_data.raw')
            
            
            j = j + 1    
    i = i + 1
    if(i==4 or i==5):
        i=6
    
  

print(df)
df.to_csv('df_bioFilm_Ldiff_abs_int.csv')


#line_plot = sns.lineplot(data=df, x="CL", y="CR", hue = "compressor", errorbar=('sd',2),err_style='bars')
#fig = line_plot.get_figure()
#fig.savefig('lineplot_CR.png')