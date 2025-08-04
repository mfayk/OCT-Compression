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


def compression_blosc(input_data,decomp_data,name,mode):
    CR=[]
    CL=[]
    cBW=[]
    dBW=[]
    psnr=[]

    for bound in np.logspace(start=-7, stop=-1, num=6):
        compressor = libpressio.PressioCompressor.from_config({
        # configure which compressor to use
        "compressor_id": "sz",
        # configure the set of metrics to be gathered
        "early_config": {
            "pressio:metric": "composite",
            "composite:plugins": ["time", "size", "error_stat"]
        },
        # configure SZ
        "compressor_config": {
            "pressio:abs": bound,
        }})

        comp_data = compressor.encode(input_data)
        decomp_data = compressor.decode(comp_data, decomp_data)
        metrics = compressor.get_metrics()

        CR.append(metrics['size:compression_ratio'])
        CL.append(bound)
        cBW.append((metrics['size:uncompressed_size']/1048576)/metrics['time:compress'])
        dBW.append((metrics['size:uncompressed_size']/1048576)/metrics['time:decompress'])
        psnr.append(metrics['error_stat:ssim'])


        #print(f"metrics={json.dumps(metrics, indent=4)}")
        json_object = json.dumps(metrics, indent=4)


    if(mode == "CR"):
        plt.plot(CL,CR, label=name)
    elif(mode == "cBW"):
        plt.plot(CL,cBW, label=name)
    elif(mode == "dBW"):
        plt.plot(CL,dBW, label=name)
    elif(mode == "psnr"):
        plt.plot(CL,psnr, label=name)




srcpath='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/';
srcpath='/scratch/mfaykus/BioFilm/'
pathexp='/scratch1/mfaykus/biofilm/ThorLabs/automated-oct-scans-acquisition-master/sample_scans/output/';
if not os.path.exists(pathexp):
     os.makedirs(pathexp)
l=glob.glob(srcpath+'*/*.raw')
l.sort()
#print(l)



#input_path = Path(__file__).parent / "/scratch/mfaykus/BioFilm/06.21.22/20220621_095224161_06.21.22 Control Processed Stack_processed.raw"

input_path=l[0]

size = (int(900/4),int(900/4),int(1024/4))
size = (2048, 1000, 250)
voxel_size=(10/size[0],10/size[1],2.23/size[2])
input_data=np.fromfile(input_path,dtype=np.int16).reshape(size)  #load it 
input_data=np.flip(input_data, axis=2)
#input_data = np.fromfile(input_path, dtype=np.uint8)

input_data = np.ascontiguousarray(input_data)
#input_data = input_data.astype(float)

#print(type(input_data))
#print(input_data.dtype)
#input_data = input_data.reshape(100,500,500)
decomp_data = input_data.copy()

#print(input_data)
#print(input_data.shape)
#print(size)
#exit(0)
CR=[]
CL=[]
cBW=[]
dBW=[]

i=0
for bound in np.logspace(start=-7, stop=-1, num=6):
    compressor = libpressio.PressioCompressor.from_config({
        # configure which compressor to use
        "compressor_id": "sz",
        # configure the set of metrics to be gathered
        "early_config": {
            "pressio:metric": "composite",
            "composite:plugins": ["time", "size", "error_stat"]
        },
        # configure SZ
        "compressor_config": {
            "pressio:abs": bound,
        }})
    #print(compressor.codec_id)
    #print(compressor.get_config())
    #print(compressor.get_compile_config()) 
    # run compressor to determine metrics
    comp_data = compressor.encode(input_data)
    decomp_data = compressor.decode(comp_data, decomp_data)
    metrics = compressor.get_metrics()
    print(f"bound={bound:1.0e}, metrics={json.dumps(metrics, indent=4)}")
    json_object = json.dumps(metrics, indent=4)
    
    
    with open("sample.json", "w") as outfile:
        json.dump(json_object, outfile)
    
    print(bound)
    if(i ==3):
        save_out = np.ndarray.tofile(comp_data,'202_zfp_comp_data.raw')
        save_out2 = np.ndarray.tofile(decomp_data,'202_zfp_decomp_data.raw')
        save_out3 = np.ndarray.tofile(input_data,'202_zfp_input_data.raw')
    i = i + 1
        

plt.xlabel("Bound") 
plt.ylabel("Compression Ratio") 


compression_blosc(input_data,decomp_data,"sz","CR")

plt.legend(['sz'])

plt.tight_layout()
plt.savefig("CR_sz.png")
plt.close()

#plt.yscale("log")
plt.xlabel("Bound") 
plt.ylabel("Compression Bandwidth: MB/s") 

compression_blosc(input_data,decomp_data,"sz","cBW")

plt.legend(['sz'])

plt.tight_layout()
plt.savefig("cBW_sz.png")
plt.close()

plt.xlabel("Bound") 
plt.ylabel("Decompression Bandwidth: MB/s") 

compression_blosc(input_data,decomp_data,"sz","dBW")

plt.legend(['sz'])

plt.tight_layout()
plt.savefig("dBW_sz.png")
plt.close()


plt.xlabel("Bound") 
plt.ylabel("PSNR") 

compression_blosc(input_data,decomp_data,"sz","psnr")

plt.legend(['sz'])

plt.tight_layout()
plt.savefig("PSNR_sz.png")
plt.close()