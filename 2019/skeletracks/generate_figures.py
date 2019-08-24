"""
Copyright (C) 2018 Alexandre Fioravante de Siqueira

This file is part of 'XXXXXX - Supplementary Material'.

'XXXXXX - Supplementary Material'
is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

'XXXXXX - Supplementary Material'
is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with 'XXXXXX -
Supplementary Material'. If not, see <http://www.gnu.org/licenses/>.
"""

from itertools import combinations
from mpl_toolkits.axes_grid1 import host_subplot
from scipy.ndimage import median_filter
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import gray2rgb
from skimage.draw import line
from skimage.filters import (threshold_otsu, threshold_yen, threshold_li,
                             threshold_isodata)
from skimage.graph import route_through_array
from skimage.io import imread
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects, skeletonize_3d
from skimage.restoration import denoise_tv_chambolle
from skimage.util import img_as_ubyte

import desiqueira2018 as ds

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import mpl_toolkits.axisartist as mpl_aa
import numpy as np
import os
import pandas as pd
import warnings

# Setting up the figures appearance.
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 30
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

# Defining some helping variables.
OFFSET = -15

LINE_WIDTH = 7
SCATTER_SIZE = 25

COLOR_OTSU = '#482878'
COLOR_YEN = '#3e4989'
COLOR_LI = '#26828e'
COLOR_ISO = '#6ece58'
COLOR_MLSS = '#fde725'

WEIGHT_FILTER = 0.05  # defining the weight for the TV Chambolle function.
SAVE_FIG_FORMAT = '.pdf'

MIN_SIZE = 25  # defining minimum size for processing a region.
TEST_REGION = 26

# Ignoring warnings.
warnings.filterwarnings('ignore')


def figure_1():
    """
    Figure 1. Photomicrographs from the test dataset, presenting fission
    tracks in (a) muscovite mica and (b) apatite samples.
    """

    img_test1 = imread('orig_figures/dur_grain1mica01.tif',
                       as_grey=True)
    img_test2 = imread('orig_figures/dur_grain1apatite01.tif',
                       as_grey=True)

    _, x_px = img_test1.shape
    x_um = _calibrate_aux(len_px=x_px)

    # checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # Figure 1(a).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(img_test1, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_01a' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 1(b).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(img_test2, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_01b' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_2():
    """
    Figure 2. Binarizing and skeletonizing the region highlighted in an
    input photomicrograph [1]. (a) Input photomicrograph. (b) Example
    highlighted region. (c) Binarizing the example region using the
    ISODATA algorithm (threshold: 133). (d) Skeletonizing the binary
    region in (c). Colormap: gray.

    [1] Image sample1_01.jpg, from the folder `orig_figures`. Available
    in the Supplementary Material.
    """

    image = imread('orig_figures/dur_grain1apatite01.tif', as_grey=True)
    img_bin = _processed_image(image)

    props = regionprops(label(img_bin))

    x_min, y_min, x_max, y_max = props[TEST_REGION].bbox

    img_orig = image[x_min:x_max, y_min:y_max]
    img_reg = props[TEST_REGION].image
    img_skel = skeletonize_3d(props[TEST_REGION].image)

    _, x_px = img_skel.shape
    x_um = _calibrate_aux(len_px=x_px)

    # checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # Figure 2(a).
    image_arrow = imread('misc/Fig01a.tif')
    _, xarr_px, _ = image_arrow.shape

    xarr_um = _calibrate_aux(len_px=xarr_px)

    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(image_arrow, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, xarr_um)

    plt.savefig('figures/Fig_02a' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 2(b).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(img_orig, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_02b' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 2(c).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(img_reg, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_02c' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    # Figure 2(d).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(img_skel, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_02d' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_4():
    """
    Figure 4. Labeling the extremity (green points) and intersection
    (blue points) pixels in the skeletonized region (Figure 2(d)).
    """

    image = imread('orig_figures/dur_grain1apatite01.tif', as_grey=True)
    img_bin = _processed_image(image)

    props = regionprops(label(img_bin))

    x_min, y_min, x_max, y_max = props[TEST_REGION].bbox
    img_skel = skeletonize_3d(props[TEST_REGION].image)
    _, x_px = img_skel.shape
    x_um = _calibrate_aux(len_px=x_px)

    px_ext, px_int = ds.pixels_interest(img_skel)

    # checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # Figure 4.
    fig = plt.figure(figsize=(9, 8))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)
    host.imshow(img_skel, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    for _, (y0_px, x0_px) in enumerate(px_ext):
        # extremity pixels
        host.scatter(x0_px, y0_px, c='g', s=SCATTER_SIZE)
    for _, (y0_px, x0_px) in enumerate(px_int):
        # intersection pixels
        host.scatter(x0_px, y0_px, c='b', s=SCATTER_SIZE)
    plt.savefig('figures/Fig_04' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_5():
    """
    Figure 5. Choosing track candidates in the region presented in
    Figure 2(b), obtaining extremity points two by two. Green pixels:
    Euclidean distance. Blue dots: route between the two extremity
    points. Yellow pixels: inner area of the region formed by Euclidean
    distance and route.
    """

    image = imread('orig_figures/dur_grain1apatite01.tif', as_grey=True)
    img_bin = _processed_image(image)
    props = regionprops(label(img_bin))

    img_skel = skeletonize_3d(props[TEST_REGION].image)
    _, x_px = img_skel.shape
    x_um = _calibrate_aux(len_px=x_px)

    # checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    px_ext, _ = ds.pixels_interest(img_skel)

    # Generating all figures at once.
    figures = ['a', 'b', 'c']

    for idx, pts in enumerate(combinations(px_ext, r=2)):
        px, py = pts

        img_aux = gray2rgb(img_skel)
        region_area = np.zeros(img_skel.shape)

        route, _ = route_through_array(~img_skel, px, py)
        distances, _, _ = ds.track_parameters(px, py, route)

        rows, cols = line(px[0], px[1],
                          py[0], py[1])
        img_aux[rows, cols] = [0, 255, 0]
        region_area[rows, cols] = True

        fig = plt.figure(figsize=(9, 8))
        host = host_subplot(111, axes_class=mpl_aa.Axes)
        plt.subplots_adjust(bottom=0.2)

        for pt in route:
            host.scatter(pt[1], pt[0], c='b', s=SCATTER_SIZE)
            region_area[pt[0], pt[1]] = True

        # extremity points.
        host.scatter(px[1], px[0], c='g', s=SCATTER_SIZE)
        host.scatter(py[1], py[0], c='g', s=SCATTER_SIZE)

        region_area = binary_fill_holes(region_area)

        for pt in route:
            region_area[pt[0], pt[1]] = False
            region_area[rows, cols] = False

        img_aux[region_area] = [255, 255, 0]

        host.imshow(img_aux, cmap='gray')
        host.axis['bottom', 'left'].toggle(all=False)

        guest = host.twiny()
        new_fixed_ax = guest.get_grid_helper().new_fixed_axis
        guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                            axes=guest,
                                            offset=(0, OFFSET))
        guest.axis['top'].toggle(all=False)
        guest.set_xlabel('$\mu m$')
        guest.set_xlim(0, x_um)

        filename = 'figures/Fig_05' + figures[idx] + SAVE_FIG_FORMAT
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    return None


def figure_6():
    """
    Figure 6. Track candidates chosen by the algorithm for the region in
    Figure 2(b). Green dots: extremity pixels. Green line: Euclidean
    distance between extremity pixels. Blue dots: route between
    extremity pixels.
    """

    image = imread('orig_figures/dur_grain1apatite01.tif', as_grey=True)
    img_bin = _processed_image(image)

    props = regionprops(label(img_bin))

    x_min, y_min, x_max, y_max = props[TEST_REGION].bbox
    img_skel = skeletonize_3d(props[TEST_REGION].image)
    _, x_px = img_skel.shape
    x_um = _calibrate_aux(len_px=x_px)

    _, trk_pts = ds.tracks_classify(img_skel)

    # Checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # Generating all figures at once.
    figures = ['a', 'b']

    for idx, pt in enumerate(trk_pts):
        fig = plt.figure(figsize=(9, 8))
        host = host_subplot(111, axes_class=mpl_aa.Axes)
        plt.subplots_adjust(bottom=0.2)

        img_rgb = gray2rgb(image[x_min:x_max, y_min:y_max])

        # calculating route and distances.
        route, _ = route_through_array(~img_skel, pt[0], pt[1])
        distances, _, _ = ds.track_parameters(pt[0], pt[1], route)

        # generating minimal distance line.
        rows, cols = line(pt[0][0], pt[0][1],
                          pt[1][0], pt[1][1])
        img_rgb[rows, cols] = [False, True, False]

        # plotting minimal distance and route.
        host.imshow(img_rgb, cmap='gray')
        host.axis['bottom', 'left'].toggle(all=False)

        guest = host.twiny()
        new_fixed_ax = guest.get_grid_helper().new_fixed_axis
        guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                            axes=guest,
                                            offset=(0, OFFSET))
        guest.axis['top'].toggle(all=False)
        guest.set_xlabel('$\mu m$')
        guest.set_xlim(0, x_um)

        for rt_pt in route:
            host.scatter(rt_pt[1], rt_pt[0], c='b', s=SCATTER_SIZE)

        # plotting extreme points.
        for p in pt:
            host.scatter(p[1], p[0], c='g', s=SCATTER_SIZE)

        filename = 'figures/Fig_06' + figures[idx] + SAVE_FIG_FORMAT
        plt.savefig(filename, bbox_inches='tight')
        plt.close()

    return None


def figure_7():
    """
    Figure 7. Visual representation of each track labeled by the
    segmentation algorithm, when using the ISODATA binarization
    (threshold: ~0.475). The numbers show how many tracks were counted
    in each region. Magenta lines: regions representing only one track.
    Green dots: extremity pixels. Green lines: Euclidean distance
    between extremity pixels. Blue paths: route between extremity
    pixels.
    """

    image = imread('orig_figures/dur_grain1apatite01.tif', as_grey=True)
    img_bin = _processed_image(image)

    _, x_px = image.shape
    x_um = _calibrate_aux(len_px=x_px)

    props = regionprops(label(img_bin))
    img_skel = skeletonize_3d(img_bin)
    rows, cols = np.where(img_skel != 0)

    img_rgb = gray2rgb(img_as_ubyte(image))
    img_rgb[rows, cols] = [255, 0, 255]

    # Checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)

    for prop in props:
        obj_info = []
        aux = skeletonize_3d(prop.image)
        trk_area, trk_px = ds.tracks_classify(aux)
        count_auto = ds.count_by_region(ds.regions_and_skel(prop.image))

        x_min, y_min, x_max, y_max = prop.bbox
        obj_info.append([prop.centroid[0],
                         prop.centroid[1],
                         str(count_auto[2][0][0])])
        for obj in obj_info:
            host.text(obj[1], obj[0], obj[2], family='monospace',
                      color='yellow', size='x-small', weight='bold')

        if trk_area is not None:
            for px in trk_px:
                route, _ = route_through_array(~aux, px[0], px[1])

                for rx in route:
                    host.scatter(y_min+rx[1], x_min+rx[0],
                                 c='b', s=SCATTER_SIZE+25)
                for p in px:
                    host.scatter(y_min+p[1], x_min+p[0],
                                 c='g', s=SCATTER_SIZE+25)

                rows, cols = line(x_min+px[0][0], y_min+px[0][1],
                                  x_min+px[1][0], y_min+px[1][1])
                img_rgb[rows, cols] = [0, 255, 0]

    host.imshow(img_rgb, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_07' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_8():
    """
    Figure 8. Track counting time for images in the test dataset,
    according to each binarization algorithm. Counting times are usually
    small (around 10^-2 s), except for MLSS, which employs wavelet
    decompositions in it (de Siqueira, 2014), being more time demanding
    than the other binarizations. Another fact would be that MLSS adds
    artificial regions to the binary image; then, further track counting
    is impaired when using this binarization.
    """

    count_time = pd.read_csv('counting_time.csv')
    count_auto = [count_time[count_time['sample'] == 'mica']['otsu'],
                  count_time[count_time['sample'] == 'mica']['yen'],
                  count_time[count_time['sample'] == 'mica']['li'],
                  count_time[count_time['sample'] == 'mica']['isodata'],
                  count_time[count_time['sample'] == 'mica']['mlss']]

    # Checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    pos = list(range(5))
    box_colors = ['#482878', '#3e4989', '#26828e', '#6ece58', '#fde725']
    flier_props = dict(marker='P', markerfacecolor='#386cb0',
                       markeredgecolor='#386cb0', linestyle='none')

    x_ticks = np.arange(5)
    x_labels = ['Otsu', 'Yen', 'Li', 'ISODATA', 'MLSS']

    fig, ax = plt.subplots(figsize=(16, 10))
    box_plot = ax.boxplot(count_auto, flierprops=flier_props,
                          positions=pos, widths=0.18)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)
    ax.set_yscale('log')

    num_boxes = len(count_auto)
    medians = list(range(num_boxes))

    for i in range(num_boxes):
        box = box_plot['boxes'][i]
        boxX, boxY = [[] for _ in range(2)]
        for j in range(5):
            boxX.append(box.get_xdata()[j])
            boxY.append(box.get_ydata()[j])
        box_coords = list(zip(boxX, boxY))
        box_polygon = mpatches.Polygon(box_coords,
                                       facecolor=box_colors[i])
        ax.add_patch(box_polygon)

        med = box_plot['medians'][i]
        medianX, medianY = [[] for _ in range(2)]
        for j in range(2):
            medianX.append(med.get_xdata()[j])
            medianY.append(med.get_ydata()[j])
            plt.plot(medianX, medianY, 'k')
            medians[i] = medianY[0]
        # overplot the sample averages with horizontal alignment
        # in the center of each box.
        plt.plot([np.average(med.get_xdata())],
                 [np.average(count_auto[i])],
                 color='w', marker='P', markeredgecolor='k')

    ax.set_xlabel('Binarization algorithm')
    ax.set_ylabel('Counting time (s)')

    plt.savefig('figures/Fig_08' + SAVE_FIG_FORMAT, bbox_inches='tight')
    plt.close()

    return None


def figure_9():
    """
    Figure 9. Comparison between manual and automatic counting with the
    proposed method in the test photomicrographs. The colored lines
    represent the data linear regression. Binarizations used: (a) Otsu,
    (b) Yen, (c) Li, (d) ISODATA, and (e) MLSS. Dashed line: 1:1 line.
    """

    manual_mica = ds.manual_counting(mineral='mica')
    manual_apatite = ds.manual_counting(mineral='apatite')
    manual = manual_mica + manual_apatite

    # Checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # Figure 9 (a).
    autootsu_mica = list(np.loadtxt('auto_count/autootsu_mica.csv',
                                    delimiter=','))
    autootsu_apatite = list(np.loadtxt('auto_count/autootsu_apatite.csv',
                                       delimiter=','))
    autootsu = autootsu_mica + autootsu_apatite

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.linspace(0, 250, 1000)
    ax.plot(manual, autootsu, marker='.', linestyle='None',
            markeredgecolor='black', color=COLOR_OTSU,
            markersize=SCATTER_SIZE+15)

    fit = np.polyfit(manual, autootsu, deg=1)
    fit_fn = np.poly1d(fit)

    ax.plot(x, fit_fn(x), linewidth=LINE_WIDTH, color=COLOR_OTSU)
    ax.plot(x, x, '--', color='k')
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')
    plt.savefig('figures/Fig_09a' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    # Figure 9 (b).
    autoyen_mica = list(np.loadtxt('auto_count/autoyen_mica.csv',
                                   delimiter=','))
    autoyen_apatite = list(np.loadtxt('auto_count/autoyen_apatite.csv',
                                      delimiter=','))
    autoyen = autoyen_mica + autoyen_apatite

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.linspace(0, 250, 1000)
    ax.plot(manual, autoyen, marker='.', linestyle='None',
            markeredgecolor='black', color=COLOR_YEN,
            markersize=SCATTER_SIZE+15)

    fit = np.polyfit(manual, autoyen, deg=1)
    fit_fn = np.poly1d(fit)

    ax.plot(x, fit_fn(x), linewidth=LINE_WIDTH, color=COLOR_YEN)
    ax.plot(x, x, '--', color='k')
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')
    plt.savefig('figures/Fig_09b' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    # Figure 9 (c).
    autoli_mica = list(np.loadtxt('auto_count/autoli_mica.csv',
                                  delimiter=','))
    autoli_apatite = list(np.loadtxt('auto_count/autoli_apatite.csv',
                                     delimiter=','))
    autoli = autoli_mica + autoli_apatite

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.linspace(0, 250, 1000)
    ax.plot(manual, autoli, marker='.', linestyle='None',
            markeredgecolor='black', color=COLOR_LI,
            markersize=SCATTER_SIZE+15)

    fit = np.polyfit(manual, autoli, deg=1)
    fit_fn = np.poly1d(fit)

    ax.plot(x, fit_fn(x), linewidth=LINE_WIDTH, color=COLOR_LI)
    ax.plot(x, x, '--', color='k')
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')
    plt.savefig('figures/Fig_09c' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    # Figure 9 (d).
    autoiso_mica = list(np.loadtxt('auto_count/autoiso_mica.csv',
                                   delimiter=','))
    autoiso_apatite = list(np.loadtxt('auto_count/autoiso_apatite.csv',
                                      delimiter=','))
    autoiso = autoiso_mica + autoiso_apatite

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.linspace(0, 250, 1000)
    ax.plot(manual, autoiso, marker='.', linestyle='None',
            markeredgecolor='black', color=COLOR_ISO,
            markersize=SCATTER_SIZE+15)

    fit = np.polyfit(manual, autoiso, deg=1)
    fit_fn = np.poly1d(fit)

    ax.plot(x, fit_fn(x), linewidth=LINE_WIDTH, color=COLOR_ISO)
    ax.plot(x, x, '--', color='k')
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')
    plt.savefig('figures/Fig_09d' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    # Figure 9 (e).
    automlss_mica = list(np.loadtxt('auto_count/automlss_mica.csv',
                                    delimiter=','))
    automlss_apatite = list(np.loadtxt('auto_count/automlss_apatite.csv',
                                       delimiter=','))
    automlss = automlss_mica + automlss_apatite

    fig, ax = plt.subplots(figsize=(16, 10))
    x = np.linspace(0, 250, 1000)
    ax.plot(manual, automlss, marker='.', linestyle='None',
            markeredgecolor='black', color=COLOR_MLSS,
            markersize=SCATTER_SIZE+15)

    fit = np.polyfit(manual, automlss, deg=1)
    fit_fn = np.poly1d(fit)

    ax.plot(x, fit_fn(x), linewidth=LINE_WIDTH, color=COLOR_MLSS)
    ax.plot(x, x, '--', color='k')
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')
    plt.savefig('figures/Fig_09e' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    return None


def figure_10():
    """
    Figure 10. Counting tracks in Figure 2(a). MLSS binarization creates
    artifacts in the resulting binary image, thus misleading the track
    counting algorithm, which counts 115 tracks. (a) MLSS binary image
    obtained from Figure 2, presenting the generated artifacts. (b)
    Results of the automatic counting algorithm. Manual counting: 54
    tracks. Automatic counting using ISODATA, Li, Otsu, and Yen
    binarizations, respectively: 41, 43, 41, and 44 tracks.
    """

    image = imread('orig_figures/dur_grain1mica01.tif', as_grey=True)

    filename = 'auto_count/mlss/dur_grain1mica01.csv'
    aux = pd.read_csv(filename)
    img_bin = binary_fill_holes(
        remove_small_objects(np.asarray(aux, dtype='bool')))

    _, x_px = img_bin.shape
    x_um = _calibrate_aux(len_px=x_px)

    # Checking if the folder 'figures' exists.
    if not os.path.isdir('./figures'):
        os.mkdir('./figures')

    # Figure 10(a).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)

    host.imshow(img_bin, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)
    plt.savefig('figures/Fig_10a' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    # Figure 10(b).
    fig = plt.figure(figsize=(12, 10))
    host = host_subplot(111, axes_class=mpl_aa.Axes)
    plt.subplots_adjust(bottom=0.2)

    props = regionprops(label(img_bin))
    img_skel = skeletonize_3d(img_bin)
    rows, cols = np.where(img_skel != 0)

    img_rgb = gray2rgb(img_as_ubyte(image))
    img_rgb[rows, cols] = [255, 0, 255]

    for prop in props:
        obj_info = []
        aux = skeletonize_3d(prop.image)
        trk_area, trk_px = ds.tracks_classify(aux)
        count_auto = ds.count_by_region(ds.regions_and_skel(prop.image))

        x_min, y_min, x_max, y_max = prop.bbox
        obj_info.append([prop.centroid[0],
                         prop.centroid[1],
                         str(count_auto[2][0][0])])
        for obj in obj_info:
            host.text(obj[1], obj[0], obj[2], family='monospace',
                      color='yellow', size='x-small', weight='bold')

        if trk_area is not None:
            for px in trk_px:
                route, _ = route_through_array(~aux, px[0], px[1])

                for rx in route:
                    host.scatter(y_min+rx[1], x_min+rx[0],
                                 c='b', s=SCATTER_SIZE+25)
                for p in px:
                    host.scatter(y_min+p[1], x_min+p[0],
                                 c='g', s=SCATTER_SIZE+25)

                rows, cols = line(x_min+px[0][0], y_min+px[0][1],
                                  x_min+px[1][0], y_min+px[1][1])
                img_rgb[rows, cols] = [0, 255, 0]

    host.imshow(img_rgb, cmap='gray')
    host.axis['bottom', 'left'].toggle(all=False)

    guest = host.twiny()
    new_fixed_ax = guest.get_grid_helper().new_fixed_axis
    guest.axis['bottom'] = new_fixed_ax(loc='bottom',
                                        axes=guest,
                                        offset=(0, OFFSET))
    guest.axis['top'].toggle(all=False)
    guest.set_xlabel('$\mu m$')
    guest.set_xlim(0, x_um)

    plt.savefig('figures/Fig_10b' + SAVE_FIG_FORMAT,
                bbox_inches='tight')
    plt.close()

    return None


def _calibrate_aux(len_px=760):
    """
    """

    calib = pd.read_csv('calibration/calibration_curve.csv')
    poly_coef = np.polyfit(x=calib['px'], y=calib['mm'], deg=1)

    ang_coef, lin_coef = poly_coef
    um = (len_px*ang_coef + lin_coef) * 1e3

    return um


def _plot_aux(image, ax=None):
    """
    """

    if ax is None:
        ax = plt.gca()

    img_skel = skeletonize_3d(image)
    rows, cols = np.where(img_skel != 0)
    img_rgb = gray2rgb(img_as_ubyte(image))
    img_rgb[rows, cols] = [255, 0, 255]

    ax.imshow(img_rgb, cmap='gray')

    props = regionprops(label(image))

    for prop in props:

        aux = skeletonize_3d(prop.image)
        trk_area, trk_px = ds.tracks_classify(aux)
        x_min, y_min, x_max, y_max = prop.bbox

        if trk_area is not None:
            for px in trk_px:
                route, _ = route_through_array(~aux, px[0], px[1])

                # plotting route.
                for rx in route:
                    ax.scatter(y_min+rx[1], x_min+rx[0],
                               c='b', s=SCATTER_SIZE+25)

                # plotting extreme points.
                for p in px:
                    ax.scatter(y_min+p[1], x_min+p[0],
                               c='g', s=SCATTER_SIZE+25)

                rows, cols = line(x_min+px[0][0], y_min+px[0][1],
                                  x_min+px[1][0], y_min+px[1][1])
                img_rgb[rows, cols] = [0, 255, 0]

    return ax


def _processed_image(image, threshold='isodata'):
    """
    """

    func_thresh = {'otsu': threshold_otsu,
                   'yen': threshold_yen,
                   'li': threshold_li,
                   'isodata': threshold_isodata,
                   }.get(threshold, threshold_isodata)

    image = denoise_tv_chambolle(image, weight=WEIGHT_FILTER)
    thresh = func_thresh(image)
    img_bin = ds.clear_rd_border(remove_small_objects(
        binary_fill_holes(image < thresh),
        min_size=MIN_SIZE))

    return img_bin


def generate_all_figures():

    import generate_figures

    print('Generating all figures...')

    for function in dir(generate_figures):
        if function.startswith('figure'):
            item = getattr(generate_figures, function)
            if callable(item):
                item()

    return None


if __name__ == '__main__':
    generate_all_figures()
