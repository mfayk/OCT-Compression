#!/usr/bin/env python


import libpressio
import argparse
import pprint


print("compressors", libpressio.supported_compressors())

print("metrics", libpressio.supported_metrics())

print("io", libpressio.supported_io())

compressor = libpressio.PressioCompressor.from_config({"compressor_id": "sz3"})

print("runtime settings")
pprint.pprint(compressor.get_config()['compressor_config'])
print("compile-time settings")
pprint.pprint(compressor.get_compile_config())