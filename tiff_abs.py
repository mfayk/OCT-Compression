#!/usr/bin/env python

from pathlib import Path
import json
import libpressio
import numpy as np
import sys
import array
import os
import math
import numpy as np
from ipywidgets import widgets,interact,IntProgress
import matplotlib.pyplot as plt
from skimage import morphology
from skimage.morphology import closing, square, reconstruction
from skimage import filters
from sklearn import preprocessing
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
    sizes = [505148*50, 481165*50, 321148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 509148*50, 509148*50, 509148*50, 509148*50]
    #sizes = [505148*50, 321148*50, 509148*50, 509148*50, 509148*50, 509148*50]
    #sizes = [505148*50, 481165*50, 509148*50, 509148*50, 509148*50, 509148*50]
    threads = [1]
    for thread in threads:
        mode_count = 0
        while mode_count < 3:
            mode = modes[mode_count]
            i = 0
            while i < len(paths):
                size = sizes[i]
                for comp in compressors:
                    j = 0
                    bounds = [math.exp(-7),math.exp(-6),math.exp(-5),math.exp(-4),math.exp(-3),math.exp(-2),math.exp(-1)]
                    #bnd = ["1e^-7", "1e^-6","1e^-5","1e^-4","1e^-3","1e^-2","1e^-1"]
                    while(j<len(bounds)):
                        bound = bounds[j]
                        
                        l=glob.glob(srcpath+paths[i])
                        l.sort()

                        
                        input_data = np.array([np.array(Image.open(fname)) for fname in l])
                        ori_data = input_data.copy()
                        print(input_data.dtype)
                        shapee = input_data.shape
                        sz1 = shapee[0]
                        sz2 = shapee[1]
                        sz3 = shapee[2]
                        
                        print(sz1)
                        print(sz2)
                        print(sz3)

                        input_data = input_data.astype(np.float32)
                        
                        
                        i_data = input_data.copy()
                        D_data = input_data.copy()
                        diff_data = input_data.copy()
                        diff_data2 = input_data.copy()
                        decomp_data = input_data.copy()
                        de_data = input_data.copy()
                        
                        bin_array = input_data.copy()
                        bin_array = bin_array.astype(np.int32)
                        
                        
                        binary = array.array('B', [0] * (50*509*1000))
                        binary2 = input_data.copy()

                        #start timer
                        start = time.time()

                        #input_data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX)
                        
                        #input_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min())
                        
                        #print(decomp_data.max())
                        #print(decomp_data.min())
                        
                        normalize_time = time.time() - start

                        
                        if(mode == "Ldiff"):
                            d1=1
                            d2=0
                            d3=0
                            while d1 < sz1:
                                while d2 < sz2:
                                    while d3 < sz3:
                                        diff_data[d1][d2][d3] = abs(input_data[d1][d2][d3]-input_data[d1-1][d2][d3])
                                        if input_data[d1][d2][d3]-input_data[d1-1][d2][d3] < 0:        
                                            binary[(d2 * sz2 * sz3) + (d3 * sz3) + d1] = 1
                                            binary2[d1][d2][d3] = 1
                                        else:
                                            binary[(d2 * sz2 * sz3) + (d3 * sz3) + d1] = 0
                                            binary2[d1][d2][d3] = 0
                                        d3 = d3 + 1
                                    d2 =d2 + 1
                                d1 = d1 + 1
                        elif(mode == "0diff"):
                            d1=1
                            d2=0
                            d3=0
                            while d1 < sz1:
                                while d2 < sz2:
                                    while d3 < sz3:
                                        diff_data[d1][d2][d3] = abs(input_data[d1][d2][d3]-input_data[0][d2][d3])
                                        if input_data[d1][d2][d3]-input_data[0][d2][d3] < 0:        
                                            binary[(d2 * sz2 * sz3) + (d3 * sz3) + d1] = 1
                                            binary2[d1][d2][d3] = 1
                                        else:
                                            binary[(d2 * sz2 * sz3) + (d3 * sz3) + d1] = 0
                                            binary2[d1][d2][d3] = 0
                                        d3 = d3 + 1
                                    d2 =d2 + 1
                                d1 = d1 + 1
                        
                        
                        temp1 = diff_data[49]
                        temp1= cv2.normalize(temp, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        path = 'tiff_data/'    
                        im = Image.fromarray(temp1)
                        im.save(path + "diff"  + '.tif')
                        imageio.imwrite(path + "diff"  + '.tif', temp1)

                        
                        

                        #d=1
                        #if(mode == "Ldiff"):
                           # diff_data2[0] = input_data[0]
                           # while d < len(l):
                                #if(input_data[d]-input_data[d-1] >= 0):
                                 #   bin_array[d] = 0
                                #else:
                                #    bin_array[d] = 1
                              #  diff_data2[d] =  (input_data[d]-input_data[d-1])
                                #path = 'diff_data/'
                                #diff_data[d] = 255 - diff_data[d]
                                #im = Image.fromarray(input_data[d].astype(np.uint8))
                                #im.save(path +  'L' + '.tif')
                                
                                #im = Image.fromarray(input_data[d-1].astype(np.uint8))
                                #im.save(path +  'L-1' + '.tif')
                                
                                #im = Image.fromarray(diff_data[d].astype(np.uint8))
                                #im.save(path +  'd' + '.tif')
                                
                                #exit()
                                
                                
                               # d=d+1
                       # elif(mode == "0diff"):
                           # diff_data2[0] = input_data[0]
                        #    while d < len(l):
                                #if(input_data[d]-input_data[d-1] >= 0):
                                #    bin_array[d] = 0
                                #else:
                                #    bin_array[d] = 1
                               # diff_data2[d] = abs(input_data[d]-input_data[0])
                               # d=d+1
                        if(mode != 'base'):
                            input_data = diff_data.copy()
                        
                        input_data = cv2.normalize(input_data, None, 0, 1, cv2.NORM_MINMAX)
                        print(mode)
                        #print(np.array_equal(diff_data2, diff_data))

                        diff_time = time.time() - start - normalize_time

                        
                        
                        #lossy portion
                        compressor1 = libpressio.PressioCompressor.from_config({
                                # configure which compressor to use
                            "compressor_id": comp,
                                # configure the set of metrics to be gathered
                                "early_config": {
                                    "pressio:metric": "composite",
                                    "sz:sz_mode":0,
                                    "composite:plugins": ["time", "size", "error_stat"]
                                },
                                # configure SZ/zfp
                                "compressor_config": {
                                    "pressio:abs": bound,
                            }})

                        comp_data1 = compressor1.encode(input_data)
                        
                        #binary1 = comp_data1.view(dtype=np.float64)
                        
                        binary1 = np.array(binary)
                        
                        #binary1 = np.concatanate(comp_data1,binary2)
                        
                        #print((binary1.dtype))
                        #print(type(comp_data1.dtype))
                        
                        #comp_data1 =np.frombuffer(comp_data1)
                        
                        print("before")
                        if mode != "base":
                            buffer_data = np.concatenate((comp_data1,binary1))
                        else:
                            buffer_data = comp_data1
                        print("after")
                        print(buffer_data.shape)
                        print(comp_data1.shape)
                        
                        #comp_data1 = np.append(comp_data1,binary)
                        
                        
                        #lossless portion
                        compressor2 = libpressio.PressioCompressor.from_config({
                                  # configure which compressor to use
                            "compressor_id": "blosc",
                                  # configure the set of metrics to be gathered
                                  "early_config": {
                                    "blosc:compressor": "zstd",
                                    "blosc:metric": "composite",
                                    "composite:plugins": ["time", "size", "error_stat", "external"]
                                    },
                                    "compressor_config": {
                                            "blosc:clevel": 9,

                                }})
                        
                        
                        print("before")
                        comp_data = compressor2.encode(buffer_data)
                        print("after")
                       
                        print(buffer_data.shape)
                        print(comp_data1.shape)
                        print(comp_data.shape)
                        print(len(binary))
                        decomp1_data = buffer_data.copy()

                        comp_time = time.time() - diff_time - normalize_time - start
                        encoding_time = time.time()- start

                        print("first decode")
                        decomp1_data = compressor2.decode(comp_data, decomp1_data)
                        print("done")
                        
                        print("decomp")
                        print(decomp1_data.shape)
                        if mode != "base":
                            decomp_data1 = decomp1_data[:-25450000]
                            binary = decomp1_data[-25450000:]
                        else:
                            decomp_data1 = decomp1_data
                        
                        print(decomp_data1.shape)
                        
                        decomp2_data = input_data.copy()
                        
                        print("second decode")
                        decomp_data = compressor1.decode(decomp_data1, decomp2_data)
                        print("done")
                        
                        
                        print("decomp")
                        print(decomp_data.shape)
                        
                        decomp_data = cv2.normalize(decomp_data, None, 0, 255, cv2.NORM_MINMAX)

                        
                        
                        if(mode == "Ldiff"):
                            d1=0
                            d2=0
                            d3=0
                            de_data[0] = decomp_data[0]
                            temp = de_data[0]
                            while d1 < sz1:
                                while d2 < sz2:
                                    while d3 < sz3:
                                        if  binary[(d2 * sz2 * sz3) + (d3 * sz3) + d1] == 1:        
                                            de_data[d1][d2][d3] = ((-de_data[d1][d2][d3])+temp[d2][d3])
                                        else:
                                            de_data[d1][d2][d3] = (de_data[d1][d2][d3]+temp[d2][d3])
                                        d3 = d3 + 1
                                    d2 =d2 + 1
                                temp = de_data[d1]
                                d1 = d1 + 1
                        elif(mode == "0diff"):
                            d1=0
                            d2=0
                            d3=0
                            while d1 < sz1:
                                while d2 < sz2:
                                    while d3 < sz3:
                                        if  binary[(d2 * sz2 * sz3) + (d3 * sz3) + d1] == 1:        
                                            de_data[d1][d2][d3] = ((-de_data[d1][d2][d3])+de_data[d1][d2][0])
                                        else:
                                            de_data[d1][d2][d3] = (de_data[d1][d2][d3]+de_data[d1][d2][0])
                                        d3 = d3 + 1
                                    d2 =d2 + 1
                                d1 = d1 + 1
                            
                        
                        
                        
                        
                        #d=1
                        #if(mode == "Ldiff"):
                            #de_data[0] = decomp_data[0]
                            #temp = de_data[0]
                         #   while d < len(l):
                                #de_data[d] =  decomp_data[d] + temp
                                #temp = de_data[d]
                         #       d=d+1

                        #elif(mode == "0diff"):
                            #de_data[0] = decomp_data[0]
                            #while d < len(l):
                                #de_data[d] =  decomp_data[d] + decomp_data[0]
                                #d=d+1

                        if(mode != 'base'):
                            decomp_data = de_data.copy()

                        D_data = decomp_data.copy()
                        
                        #D_data = cv2.normalize(decomp_data, None, 0, 255, cv2.NORM_MINMAX, cv2.CV_8U)
                        #D_data = (decomp_data - decomp_data.min())/ (decomp_data.max() - decomp_data.min()) * 255
                        
                        de_comp_time = time.time() - encoding_time - start
                        end = time.time() - start
                        
                        #D_data = decomp_data.copy()
                        
                        print("The time of execution of above program is :",(end) * 10**3, "ms")
                        metrics = compressor2.get_metrics()
                        D_data = D_data.astype(np.uint8)
                        (ssim, diff) = structural_similarity(i_data, D_data, data_range=255, full=True)


                        diff = (diff * 255).astype("uint8")

                        print("SSIM: {}".format(ssim))
#get size from libpressio

                        df.loc[len(df)] = [paths[i], bound, comp, ssim, metrics['error_stat:psnr'] , (size/encoding_time)/1000000, (size/de_comp_time)/1000000, size/metrics['size:compressed_size'],normalize_time,diff_time,comp_time,encoding_time,de_comp_time,end,mode,thread]


                        
                        save = 1

                        if j == 3:
                            save = 0

                        if save == 0:
                            num=0
                            print(mode)
                            for img in D_data:
                                path = 'tiff_data/'
                                im = Image.fromarray(img)
                                im.save(path + str(num)  + '.tif')
                                imageio.imwrite(path + str(num)  + '.tif', img)
                                num +=1
                                
                            
                            
                            
                            exit(0)
                            
                            num=0
                            


                            #exit(0)

                        j = j + 1




                i = i + 1
            mode_count = mode_count + 1
    return df




paths = ['10.20.22/ATLC3_Trial1/*.tif', '11.17.22/ATLC8b/*.tif', '3.10.23/ATLC3/*.tif', '3.10.23/HB/*.tif', '3.15.23/ATLC3.3.15.23/*.tif', 
'3.15.23/HB2.3.15.23/*.tif', '3.29.23/ATLC3/Static/*.tif', '3.29.23/ATLC3/Transfer/*.tif', '3.29.23/HB2/Static/*.tif', '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

#paths = ['10.20.22/ATLC3_Trial1/*.tif', '11.17.22/ATLC8b/*.tif',  '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

#paths = ['10.20.22/ATLC3_Trial1/*.tif',  '3.10.23/ATLC3/*.tif',  '9.30.22/ATCL8b.Day3.Trial1/*.tif', '9.30.22/Control/*.tif', '9.30.22/H2.Day3.Trial1/*.tif', '9.30.22/H2.Day3.Trial2/*.tif']

#sizes = [505148*50, 481165*50, 321148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 505148*50, 509148*50, 509148*50, 509148*50, 509148*50]

#sizes2 = [505148*50, 321148*50, 509148*50, 509148*50, 509148*50, 509148*50]

#sizes2 = [505148*50, 481165*50, 509148*50, 509148*50, 509148*50, 509148*50]

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
compressors = ['sz', 'zfp']
compressors = ['sz']

df = pd.DataFrame({
            'filename':[],
            'bound':[],
            'compressor':[],
            'ssim':[],
            'psnr':[],
            'cBW':[],
            'dBW':[],
            'CR':[],
            'normalize_time':[],
            'diff_time':[],
            'comp_time':[],
            'encoding_time':[],
            'de_comp_time':[],
            'total_time':[],
            'diff':[],
            'thread':[]})


mode = [ "0diff","base", "Ldiff"]

data = compression(srcpath, paths, compressors, mode)
data.to_csv('data/bioFilm_tiff_lossy_abs.csv')
