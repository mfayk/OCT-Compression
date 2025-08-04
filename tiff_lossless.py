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

def compression(srcpath, paths, compressor, modes):
    #threads = [1,2,4,8,16,32]
    threads = [1]
    
    sizes = [505148*50, 481165*50, 321148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 509148*50, 509148*50, 509148*50, 509148*50]
    
    for thread in threads:
        mode_count = 0
        while mode_count < 3:
            mode = modes[mode_count]
            i = 0
            while i < len(paths):
                size = sizes[i]
                for comp in compressors:
                    j = 0
                    while j < 9:
                        l=glob.glob(srcpath+paths[i])
                        l.sort()


                        input_data = np.array([np.array(Image.open(fname)) for fname in l])
                        ori_data = input_data.copy()
                        input_data = input_data.astype(np.int32)
                        i_data = input_data.copy()

                        #start timer
                        start = time.time()

                        #data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX)

                        #input_data = data.copy()
                        diff_data = input_data.copy()
                        decomp_data = input_data.copy()
                        de_data = input_data.copy()

                        normalize_time = time.time() - start


                        d=1
                        if(mode == "Ldiff"):
                            diff_data[0] = input_data[0]
                            while d < len(l):
                                diff_data[d] =  (input_data[d]-input_data[d-1])
                                #path = 'diff_data/'
                                #diff_data[d] = 255 - diff_data[d]
                                #im = Image.fromarray(input_data[d].astype(np.uint8))
                                #im.save(path +  'L' + '.tif')
                                
                                #im = Image.fromarray(input_data[d-1].astype(np.uint8))
                                #im.save(path +  'L-1' + '.tif')
                                
                                #im = Image.fromarray(diff_data[d].astype(np.uint8))
                                #im.save(path +  'd' + '.tif')
                                
                                #exit()
                                
                                
                                d=d+1
                        elif(mode == "0diff"):
                            diff_data[0] = input_data[0]
                            while d < len(l):
                                diff_data[d] = (input_data[d]-input_data[0])
                                d=d+1
                        if(mode != 'base'):
                            input_data = diff_data.copy()


                        diff_time = time.time() - start - normalize_time
                        
                        compressor = libpressio.PressioCompressor.from_config({
                                  # configure which compressor to use
                            "compressor_id": "blosc",
                                  # configure the set of metrics to be gathered
                                  "early_config": {
                                    "blosc:compressor": comp,
                                    "blosc:metric": "composite",
                                      "pressio:nthreads":thread,
                                    "composite:plugins": ["time", "size", "error_stat", "external"]
                                    },
                                    "compressor_config": {
                                            "blosc:clevel": j,

                                }})

                        comp_data = compressor.encode(input_data)
                        compressed_data = comp_data.copy()

                        
                        name = "lossless" + str(j)
                        np.save(name,compressed_data)
                        
                        

                        comp_time = time.time() - diff_time - normalize_time - start
                        encoding_time = time.time()- start

                        decomp_data = compressor.decode(comp_data, decomp_data)

                        d=1
                        if(mode == "Ldiff"):
                            de_data[0] = decomp_data[0]
                            temp = de_data[0]
                            while d < len(l):
                                de_data[d] =  decomp_data[d] + temp
                                temp = de_data[d]
                                d=d+1

                        elif(mode == "0diff"):
                            de_data[0] = decomp_data[0]
                            while d < len(l):
                                de_data[d] =  decomp_data[d] + decomp_data[0]
                                d=d+1

                        if(mode != 'base'):
                            decomp_data = de_data.copy()


                        D_data = decomp_data.copy()
                        #D_data = cv2.normalize(decomp_data, None, 0, 255, cv2.NORM_MINMAX)


                        de_comp_time = time.time() - encoding_time - start
                        end = time.time() - start
                        print("The time of execution of above program is :",(end) * 10**3, "ms")
                        metrics = compressor.get_metrics()


                        df.loc[len(df)] = [(size/encoding_time)/1000000,size/metrics['size:compressed_size']]


                        D_data = D_data.astype(np.uint8)
                        save = 1

                        if j == 3:
                            save = 1

                        if save == 0:
                            num=0
                            print(mode)
                            for img in D_data:
                                path = 'tiff_data/'
                                im = Image.fromarray(img)
                                im.save(path + str(num)  + '.tif')
                                #imageio.imwrite(path + str(num)  + '.tif', img)
                                num +=1
                            num=0
                            print(compressed_data.size)
                            print("sys size")
                            print(sys.getsizeof(compressed_data))
                            path = 'comp_data/'
                            im = Image.fromarray(compressed_data)
                            im.save(path + '1'  + '.tif')
                            print(metrics['size:compression_ratio'])
                            print(paths[i])

                            #exit(0)

                        j = j + 1




                i = i + 1
            mode_count = mode_count + 1
            return df
            continue
    return df


#paths = ['10.20.22/ATLC3_Trial1/*.tif', '11.17.22/ATLC8b/*.tif', '3.10.23/ATLC3/*.tif', '3.10.23/HB/*.tif', '3.15.23/ATLC3.3.15.23/*.tif', 
#'3.15.23/HB2.3.15.23/*.tif', '3.29.23/ATLC3/Static/*.tif', '3.29.23/ATLC3/Transfer/*.tif', '3.29.23/HB2/Static/*.tif', '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

paths = ['10.20.22/ATLC3_Trial1/*.tif']

#paths = ['10.20.22/ATLC3_Trial1/*.tif', '11.17.22/ATLC8b/*.tif', '3.10.23/ATLC3/*.tif', '3.10.23/HB/*.tif', '3.15.23/HB2.3.15.23/*.tif', '3.29.23/ATLC3/Static/*.tif', '3.29.23/ATLC3/Transfer/*.tif', '3.29.23/HB2/Static/*.tif', '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

#paths = ['10.20.22/ATLC3_Trial1/*.tif',  '3.10.23/ATLC3/*.tif',  '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

#srcpath='/scratch1/mfaykus/automated-oct-scans-acquisition/sample_scans/';
srcpath='/scratch/mfaykus/BioFilm/'
pathexp='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/output/';
if not os.path.exists(pathexp):
     os.makedirs(pathexp)
#l=glob.glob(srcpath+'10.20.22/ATLC3_Trial1/*.tif')
#l=glob.glob(srcpath+'11.17.22/ATLC8b/*.tif')
#l=glob.glob(srcpath+'3.10.23/ATLC3/*.tif')
#l=glob.glob(srcpath+'3.10.23/HB/*.tif')
#l=glob.glob(srcpath+'3.15.23/HB2.3.15.23/*.tif')
#l=glob.glob(srcpath+'3.29.23/ATLC3/Static/*.tif')
#l=glob.glob(srcpath+'3.29.23/ATLC3/Transfer/*.tif')
#l=glob.glob(srcpath+'3.29.23/HB2/Static/*.tif')
#l=glob.glob(srcpath+'9.30.22/ATCL8b.Day3.Trial1/*.tif')
#l=glob.glob(srcpath+'9.30.22/Control/*.tif')
#l=glob.glob(srcpath+'9.30.22/H2.Day3.Trial1/*.tif')
#l=glob.glob(srcpath+'9.30.22/H2.Day3.Trial2/*.tif')

#compressors = ['zstd','blosclz','lz4','lz4hc','zlib']

compressors = ['zstd']

df = pd.DataFrame({
            'cBW':[],
            'CR':[]
})


#mode = ["Ldiff", "0diff", "base"]
mode = ["0diff"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv('data/bioFilm_tiff_lossless_abs.csv')
