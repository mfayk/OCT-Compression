# OCT-Compression
Lossy / Lossless compression of OCT biofilm images with libpressio


![alt text](https://github.com/mfayk/OCT-Compression/blob/main/bscan.jpg)


Lossy and Lossless Compression for BioFilm Optical Coherence Tomography (OCT)

https://dl.acm.org/doi/fullHtml/10.1145/3624062.3625125

Optical Coherence Tomography (OCT) is a fast and non-destructive technology for bacterial biofilm imaging. However, OCT generates approximately 100 GB per flow cell, which complicates storage and data sharing. Data reduction reduces data complications by reducing overhead and the amount of data transferred. This work leverages the similarities between layers of OCT images to minimize data in order to improve compression. This paper evaluates 5 lossless and 2 lossy state-of-the-art compressors as well as 2 pre-processing techniques to reduce OCT data. Reduction techniques are evaluated to determine which compressor has the most significant compression ratio while maintaining a strong bandwidth and minimal image distortion. Results show SZ with frame before pre-processing is able to achieve the highest CR of 204.6 × on its higher error bounds. The maximum compression bandwidth SZ on higher error bounds is ∼ 41MB/s, for decompression bandwidth, it is able to outperform ZFP achieving ∼ 67MB/s.
