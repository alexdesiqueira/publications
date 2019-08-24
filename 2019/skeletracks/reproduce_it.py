"""
"""

from glob import glob
from itertools import combinations, permutations, product
from matplotlib import rcParams
from matplotlib.ticker import MultipleLocator
from mpl_toolkits.axes_grid1 import make_axes_locatable
from scipy import ndimage
from scipy.ndimage.morphology import binary_fill_holes
from scipy.stats import norm
from skimage.color import gray2rgb
from skimage.draw import line
from skimage.exposure import rescale_intensity
from skimage.graph import route_through_array
from skimage.io import ImageCollection, imread, imsave
from skimage.util import img_as_float, img_as_ubyte
from skimage.filters import (threshold_otsu, threshold_yen,
                             threshold_li, threshold_isodata)
from skimage.measure import regionprops, label, compare_ssim
from skimage.morphology import (remove_small_objects, skeletonize,
                                skeletonize_3d)
from skimage.restoration import denoise_tv_chambolle
from skimage.segmentation import clear_border

from supmat_support import *

import desiqueira2018 as ds

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import os
import statistics as stats

plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 15
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = 1.5*plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']


# Obtaining the image collection:
# * First, we use ImageCollection() to read all images in the folder
#   orig_figures. After that, we print the number of images in this
#   dataset.

def imread_convert(image):
    """Support function for ImageCollection. Converts the input image to
    gray.
    """

    return imread(image, as_grey=True)


files_mica = 'orig_figures/*mica*.tif'
files_apatite = 'orig_figures/*apatite*.tif'

imgset_mica = ImageCollection(load_pattern=files_mica,
                              load_func=imread_convert)
imgset_apatite = ImageCollection(load_pattern=files_apatite,
                                 load_func=imread_convert)

print('Number of images on dataset:',
      len(imgset_mica), 'mica,',
      len(imgset_apatite), 'apatite.')


# Binarizing images
# * Here we binarize all images using different algorithms: Otsu, Yen,
#   Li, ISODATA, triangle, MLSS.
# * We also perform some cleaning actions:
#  ** remove_small_objects(), in its default settings, removes objects
#     with an area smaller than 64 px.
#  ** binary_fill_holes() fills holes contained in objects.
#  ** clear_rd_border() removes objects touching the lower and right
#     borders. These objects could not be identified with precision.
# * After that, we present the results of these actions on the first
#   images from the set.

def binarize_imageset(image_set):

    imgbin_otsu, imgbin_yen, imgbin_li, imgbin_iso, imgbin_tri, imgbin_mlss = [[] for _ in range(6)]

    for idx, img in enumerate(image_set):
        # Filtering
        #img = ndimage.median_filter(img, size=(7, 7))
        img = denoise_tv_chambolle(img, weight=0.05)
        #img = rescale_intensity(img, in_range=(0, 0.5))
        # Otsu
        imgbin_otsu.append(binarize_imageset_aux(img < threshold_otsu(img)))
        # Yen
        imgbin_yen.append(binarize_imageset_aux(img < threshold_yen(img)))
        # Li
        imgbin_li.append(binarize_imageset_aux(img < threshold_li(img)))
        # ISODATA
        imgbin_iso.append(binarize_imageset_aux(img < threshold_isodata(img)))
        # MLSS
        aux = pd.read_csv('auto_count/mlss/imgbin_mlss' + str(idx+1) + '.csv')
        imgbin_mlss.append(binarize_imageset_aux(np.asarray(aux, dtype='bool')))

    return imgbin_otsu, imgbin_yen, imgbin_li, imgbin_iso, imgbin_mlss


def binarize_imageset_aux(image):
    
    return ds.clear_rd_border(remove_small_objects(binary_fill_holes(image)))
