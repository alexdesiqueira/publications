"""
Copyright (C) 2017 Alexandre Fioravante de Siqueira

This file is part of 'Segmentation of nearly isotropic overlapped
tracks in photomicrographs using successive erosions as watershed
markers - Supplementary Material'.

'Segmentation of nearly isotropic overlapped tracks in photomicrographs
using successive erosions as watershed markers - Supplementary Material'
is free software: you can redistribute it and/or modify it under
the terms of the GNU General Public License as published by the
Free Software Foundation, either version 3 of the License, or
(at your option) any later version.

'Segmentation of nearly isotropic overlapped tracks in photomicrographs
using successive erosions as watershed markers - Supplementary Material'
is distributed in the hope that it will be useful,
but WITHOUT ANY WARRANTY; without even the implied warranty of
MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
GNU General Public License for more details.

You should have received a copy of the GNU General Public License
along with 'Segmentation of nearly isotropic overlapped tracks in
photomicrographs using successive erosions as watershed markers -
Supplementary Material'. If not, see <http://www.gnu.org/licenses/>.
"""

from glob import glob
from itertools import chain, product
from matplotlib import mlab
from matplotlib.animation import ArtistAnimation
from scipy.ndimage.morphology import (binary_fill_holes,
                                      distance_transform_edt)
from scipy.stats import norm
from skimage.filters import threshold_isodata
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import binary_erosion, disk, watershed
from skimage.segmentation import clear_border
from string import ascii_lowercase

import desiqueira2017 as ds

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
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

default_fontsize = 15

# Ignoring warnings.
warnings.filterwarnings('ignore')


def figure_3():
    """
    Figure 3. WUSEM algorithm application in an input image. First, the
    image is binarized using the ISODATA threshold. Then, the binary
    image is eroded using initial radius (r0) and iterative radius (∆r)
    equal to 1 and 1 respectively, to ease visualization. The process
    continues until the eroded image has regions in it. All erosions are
    summed, and the result is labeled; then, regions with area smaller
    than 64 px are excluded. Each labeled track receives a color from
    the nipy_spectral colormap. Finally, the function enumerate_objects()
    is used to number the found tracks. Final results are shown to r0
    and ∆r equal to 10 and 4, respectively. Animation also available at
    <https://youtu.be/CtIOxhNISW8>.
    """

    image_animation = []

    img_orig = imread(('figures/orig_figures/dataset_01/Kr-78_4,5min/'
                       'K90_incid/K90_incid4,5min_3.bmp'), as_grey=True)

    fig = plt.figure(figsize=(15, 10))

    # 1st image: original photomicrograph.
    curr_frame = plt.imshow(img_orig, cmap='gray')
    for i in range(10):
        if i+1 < 10:
            fname = 'misc/fig_3-frames/frame0' + str(i+1) + '.eps'
        else:
            fname = 'misc/fig_3-frames/frame' + str(i+1) + '.eps'
        plt.savefig(fname=fname, bbox_inches='tight')

    # 2nd image: binary image.
    aux = img_orig < threshold_isodata(img_orig)
    image = binary_fill_holes(aux)
    curr_frame = plt.imshow(image, cmap='gray', animated=True)
    for i in range(10):
        fname = 'misc/fig_3-frames/frame' + str(i+11) + '.eps'
        plt.savefig(fname=fname, bbox_inches='tight')

    rows, cols = image.shape
    distance = distance_transform_edt(image)

    # following images: erosions.
    initial_radius, delta_radius = 1, 1
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius

    counter = 0

    while True:
        erod_aux = binary_erosion(image, selem=disk(curr_radius))
        curr_frame = plt.imshow(erod_aux, cmap='gray', animated=True)

        fname = 'misc/fig_3-frames/frame' + str(counter+21) + '.eps'
        plt.savefig(fname=fname, bbox_inches='tight')

        if erod_aux.min() == erod_aux.max():
            break

        markers = label(erod_aux)
        curr_labels = watershed(-distance,
                                markers,
                                mask=image)
        img_labels += curr_labels
        curr_radius += delta_radius

        counter += 1

    # following image: chosen segmentation.
    initial_radius, delta_radius = 10, 4
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius

    while True:
        erod_aux = binary_erosion(image, selem=disk(curr_radius))

        if erod_aux.min() == erod_aux.max():
            break

        markers = label(erod_aux)
        curr_labels = watershed(-distance,
                                markers,
                                mask=image)
        img_labels += curr_labels
        curr_radius += delta_radius

    # next image: colored image.
    img_labels = ds.clear_rd_border(label(img_labels))
    curr_frame = plt.imshow(img_labels, cmap='nipy_spectral',
                            animated=True)
    for i in range(10):
        fname = 'misc/fig_3-frames/frame' + str(i+57) + '.eps'
        plt.savefig(fname=fname, bbox_inches='tight')

    # Figure 3: saving the video thumbnail.
    plt.imshow(img_labels, cmap='nipy_spectral', animated=True)
    plt.savefig('Fig_3-thumbnail.eps', bbox_inches='tight')

    # last image: numbered image.
    img_number = ds.enumerate_objects(img_orig,
                                      img_labels,
                                      font_size=30)

    curr_frame = plt.imshow(img_number, animated=True)
    for i in range(10):
        fname = 'misc/fig_3-frames/frame' + str(i+67) + '.eps'
        plt.savefig(fname=fname, bbox_inches='tight')

    # Figure 3.
    os.system('convert -delay 20 misc/fig_3-frames/frame*.eps Fig_3-media.mp4')

    return None


def figure_6():
    """
    Figure 6. Input photomicrograph binarized using the ISODATA threshold
    (threshold = 128) and region filling.
    """

    image = imread(('figures/orig_figures/dataset_01/Kr-78_4,5min/'
                    'K90_incid/K90_incid4,5min_3.bmp'), as_grey=True)

    img_bin = binary_fill_holes(image < threshold_isodata(image))

    # Figure 6.
    plt.figure(figsize=(15, 10))
    plt.imshow(img_bin, cmap='gray')
    plt.savefig('Fig_6.eps', bbox_inches='tight')

    return None


def figure_7():
    """
    Figure 7. Processing an input image using WUSEM. Each line presents
    a segmentation step. Left column: original binary image (red) and
    erosion obtained according to the variation of delta_radius (white).
    Center column: erosion results labeled, used as markers. Right column:
    watershed results when using generated markers. Parameters:
    initial_radius = 10; delta_radius = 4.
    """

    initial_radius, delta_radius, counter = (10, 4, 1)
    subfig = list(ascii_lowercase[:7])

    image = imread(('figures/orig_figures/dataset_01/Kr-78_4,5min/'
                    'K90_incid/K90_incid4,5min_3.bmp'), as_grey=True)
    thresh = threshold_isodata(image)
    img_bin = binary_fill_holes(image < thresh)

    rows, cols = image.shape
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius
    distance = distance_transform_edt(img_bin)

    while True:
        str_el = disk(curr_radius)
        erod_aux = binary_erosion(img_bin, selem=str_el)

        if erod_aux.min() == erod_aux.max():
            break

        markers = label(erod_aux)
        curr_labels = watershed(-distance, markers, mask=img_bin)

        # preparing for another loop.
        img_labels += curr_labels
        curr_radius += delta_radius

        # generating all figures at once.
        erod_diff = np.zeros((rows, cols, 3))
        erod_diff[:, :, 0][img_bin] = 1
        erod_diff[:, :][erod_aux] = [1, 1, 1]

        plt.figure(figsize=(15, 10))
        plt.imshow(erod_diff)
        plt.savefig('Fig_7' + subfig[counter-1] + 'l.eps',
                    bbox_inches='tight')

        plt.figure(figsize=(15, 10))
        plt.imshow(markers, cmap='nipy_spectral')
        plt.savefig('Fig_7' + subfig[counter-1] + 'c.eps',
                    bbox_inches='tight')

        plt.figure(figsize=(15, 10))
        plt.imshow(curr_labels, cmap='nipy_spectral')
        plt.savefig('Fig_7' + subfig[counter-1] + 'r.eps',
                    bbox_inches='tight')

        counter += 1

    return None


def figure_8():
    """
    Figure 8. (a) Labels generated from tracks in Figure 4 using the
    WUSEM algorithm. (b) Tracks in (a) enumerated using enumerate_objects().
    Tracks in the lower or right corners are not counted, according to
    the “lower right corner” method. Parameters for WUSEM algorithm:
    initial_radius = 10, delta_radius = 4. Colormaps: (a) nipy_spectral,
    (b) gray.
    """

    image = imread(('figures/orig_figures/dataset_01/Kr-78_4,5min/'
                    'K90_incid/K90_incid4,5min_3.bmp'), as_grey=True)

    img_bin = binary_fill_holes(image < threshold_isodata(image))

    # Figure 8 (a).
    img_labels, _, _ = ds.segmentation_wusem(img_bin,
                                             initial_radius=10,
                                             delta_radius=4)
    plt.figure(figsize=(15, 10))
    plt.imshow(ds.clear_rd_border(img_labels), cmap='nipy_spectral')
    plt.savefig('Fig_8a.eps', bbox_inches='tight')

    # Figure 8 (b).
    img_numbers = ds.enumerate_objects(image,
                                       ds.clear_rd_border(img_labels),
                                       font_size=25)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_numbers, cmap='gray')
    plt.savefig('Fig_8b.eps', bbox_inches='tight')

    return None


def figure_9():
    """
    Figure 9. Manual counting mean (top of the blue bar; values on the
    right) for each sample and automatic counting results with mean within
    (orange points) and outside (gray points) the tolerance interval (blue
    bar) for the first dataset.
    """

    # defining some helping variables.
    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']
    plot_where = {'0': 10, '20': 20, '30': 30, '40': 40, '50': 50,
                  '60': 60, '70': 70, '80': 80, '90': 90}
    ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]
    man_color, wusem_color = ('#386cb0', '#fdc086')

    # defining mean tolerance used in the paper.
    tol = 2

    # Figure 9 (a).
    man_count = pd.read_excel(('counting/manual_count/'
                               'manual_dataset01_Kr-78_4,5min_incid.xls'))
    wusem_count = pd.read_csv(('counting/wusem_count/'
                               'wusem_dataset01_Kr-78_4,5min_incid.txt'))
    manual, wusem, mean_man = [{} for _ in range(3)]

    for idx, folder in enumerate(folders):
        manual[samples[idx]] = man_count[man_count['folder'] == folder]
        wusem[samples[idx]] = wusem_count[wusem_count['folder'] == folder]

    # calculating the means for manual counting,and obtaining
    # best candidates for initial_radius and delta_radius.
    for key, val in manual.items():
        mean_man[key] = val.manual_count.mean()

    fig, ax = plt.subplots(figsize=(15, 10))

    for key, val in mean_man.items():
        ax.fill_between(np.arange(plot_where[key]-1,
                                  plot_where[key]+2),
                        y1=val, y2=val-tol,
                        facecolor=man_color)
        ax.annotate(np.round(val, decimals=1),
                    xy=(plot_where[key]+1.2, val), color=man_color)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        for key, val in wusem.items():
            aux = val.auto_count[(val.initial_radius == i) &
                                 (val.delta_radius == j)].mean()
            if 0 < (mean_man[key] - aux) < tol:
                ax.scatter(plot_where[key],
                           aux,
                           color=wusem_color,
                           edgecolor='k')
            else:
                ax.scatter(plot_where[key],
                           aux,
                           color='0.8',
                           edgecolor='k')

    ax.set_xticklabels(('K0', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70',
                        'K80', 'K90'))
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean counting')
    ax.set_xticks(ticks)
    ax.set_xlim([5, 100])
    ax.set_ylim([-1, 40])

    plt.savefig('Fig_9a.eps', bbox_inches='tight')

    # Figure 9 (b).
    man_count = pd.read_excel(('counting/manual_count/'
                               'manual_dataset01_Kr-78_8,5min_incid.xls'))
    wusem_count = pd.read_csv(('counting/wusem_count/'
                               'wusem_dataset01_Kr-78_8,5min_incid.txt'))
    manual, wusem, mean_man = [{} for _ in range(3)]

    for idx, folder in enumerate(folders):
        manual[samples[idx]] = man_count[man_count['folder'] == folder]
        wusem[samples[idx]] = wusem_count[wusem_count['folder'] == folder]

    for key, val in manual.items():
        mean_man[key] = val.manual_count.mean()

    fig, ax = plt.subplots(figsize=(15, 10))

    for key, val in mean_man.items():
        ax.fill_between(np.arange(plot_where[key]-1,
                                  plot_where[key]+2),
                        y1=val, y2=val-tol,
                        facecolor=man_color)
        ax.annotate(np.round(val, decimals=1),
                    xy=(plot_where[key]+1.2, val), color=man_color)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        for key, val in wusem.items():
            aux = val.auto_count[(val.initial_radius == i) &
                                 (val.delta_radius == j)].mean()
            if 0 < (mean_man[key] - aux) < tol:
                ax.scatter(plot_where[key],
                           aux,
                           color=wusem_color,
                           edgecolor='k')
            else:
                ax.scatter(plot_where[key],
                           aux,
                           color='0.8',
                           edgecolor='k')

    ax.set_xticklabels(('K0', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70',
                        'K80', 'K90'))
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean counting')
    ax.set_xticks(ticks)
    ax.set_xlim([5, 100])
    ax.set_ylim([-1, 40])

    plt.savefig('Fig_9b.eps', bbox_inches='tight')

    return None


def figure_10():
    """
    Figure 10. Comparison between manual and automatic counting for 4.5
    min etching samples. White: manual counting. Light gray: classic
    watershed counting. Dark gray: H-watershed counting. Red line:
    distribution median. White signal: distribution mean.

    Notes
    -----

    1. Based on the example available at:
    http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
    2. Colors extracted from the 'viridis' colormap. Code used:

    >>> from pylab import *
    >>> cmap = cm.get_cmap('viridis', 10)
    >>> for i in range(cmap.N):
    ...    rgb = cmap(i)[:3]
    ...    print(matplotlib.colors.rgb2hex(rgb))

    Code based on the example given in the best answer at:
    https://stackoverflow.com/questions/3016283/\
    create-a-color-generator-from-given-colormap-in-matplotlib
    """

    man_count = pd.read_excel(('counting/manual_count/'
                               'manual_dataset01_Kr-78_4,5min_incid.xls'))
    wusem_count = pd.read_csv(('counting/wusem_count/'
                               'wusem_dataset01_Kr-78_4,5min_incid.txt'))
    water_count = pd.read_csv(('counting/water_count/'
                               'water_dataset01_Kr-78_4,5min_incid.txt'))
    hwater_count = pd.read_csv(('counting/hwater_count/'
                                'hwater_dataset01_Kr-78_4,5min_incid.txt'))

    radii_best = (10, 20)
    seed_best = 5

    _aux_figs_10_11_(man_count, wusem_count, water_count, hwater_count,
                    radii_best, seed_best)

    for file in glob('Fig-*.eps'):
        os.rename(file, file[:3] + '_10' + file[3:])

    return None


def figure_11():
    """
    Figure 11. Comparison between manual and automatic counting for 8.5
    min etching samples. White: manual counting. Light gray: classic
    watershed counting. Dark gray: H-watershed counting. Red line:
    distribution median. White signal: distribution mean.

    Notes
    -----

    1. Based on the example available at:
    http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
    2. Colors extracted from the 'viridis' colormap. Code used:

    >>> from pylab import *
    >>> cmap = cm.get_cmap('viridis', 10)
    >>> for i in range(cmap.N):
    ...    rgb = cmap(i)[:3]
    ...    print(matplotlib.colors.rgb2hex(rgb))

    Code based on the example given in the best answer at:
    https://stackoverflow.com/questions/3016283/\
    create-a-color-generator-from-given-colormap-in-matplotlib
    """

    man_count = pd.read_excel(('counting/manual_count/'
                               'manual_dataset01_Kr-78_8,5min_incid.xls'))
    wusem_count = pd.read_csv(('counting/wusem_count/'
                               'wusem_dataset01_Kr-78_8,5min_incid.txt'))
    water_count = pd.read_csv(('counting/water_count/'
                               'water_dataset01_Kr-78_8,5min_incid.txt'))
    hwater_count = pd.read_csv(('counting/hwater_count/'
                                'hwater_dataset01_Kr-78_8,5min_incid.txt'))

    radii_best = (10, 14)  # best WUSEM parameters for 8.5 min
    seed_best = 5  # best H-watershed seed for 8.5 min

    _aux_figs_10_11_(man_count, wusem_count, water_count, hwater_count,
                    radii_best, seed_best)

    for file in glob('Fig-*.eps'):
        os.rename(file, file[:3] + '_11' + file[3:])

    return None


def figure_12():
    """
    Figure 12. Comparison between manual and automatic counting for (a)
    4.5 min etching samples and (b) 8.5 min etching samples. Dashed:
    1:1 line. Cyan, light gray and dark gray lines: regression for the
    WUSEM, classic watershed and H-watershed counting data, respectively.

    Notes
    -----

    1. Based on the example available at:
    http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
    2. Colors extracted from the 'viridis' colormap. Code used:

    >>> from pylab import *
    >>> cmap = cm.get_cmap('viridis', 10)
    >>> for i in range(cmap.N):
    ...    rgb = cmap(i)[:3]
    ...    print(matplotlib.colors.rgb2hex(rgb))

    Code based on the example given in the best answer at:
    https://stackoverflow.com/questions/3016283/\
    create-a-color-generator-from-given-colormap-in-matplotlib
    """

    # Figure 12 (a).
    man_count = pd.read_excel(('counting/manual_count/'
                               'manual_dataset01_Kr-78_4,5min_incid.xls'))
    wusem_count = pd.read_csv(('counting/wusem_count/'
                               'wusem_dataset01_Kr-78_4,5min_incid.txt'))
    water_count = pd.read_csv(('counting/water_count/'
                               'water_dataset01_Kr-78_4,5min_incid.txt'))
    hwater_count = pd.read_csv(('counting/hwater_count/'
                                'hwater_dataset01_Kr-78_4,5min_incid.txt'))

    _aux_fig_12_(man_count, wusem_count, water_count, hwater_count,
                 radii_best=(10, 20), seed_best=5)

    for file in glob('Fig_12.eps'):
        os.rename(file, file[:6] + 'a' + file[6:])

    # Figure 12 (b).
    man_count = pd.read_excel(('counting/manual_count/'
                               'manual_dataset01_Kr-78_8,5min_incid.xls'))
    wusem_count = pd.read_csv(('counting/wusem_count/'
                               'wusem_dataset01_Kr-78_8,5min_incid.txt'))
    water_count = pd.read_csv(('counting/water_count/'
                               'water_dataset01_Kr-78_8,5min_incid.txt'))
    hwater_count = pd.read_csv(('counting/hwater_count/'
                                'hwater_dataset01_Kr-78_8,5min_incid.txt'))

    _aux_fig_12_(man_count, wusem_count, water_count, hwater_count,
                 radii_best=(10, 14), seed_best=5)

    for file in glob('Fig_12.eps'):
        os.rename(file, file[:6] + 'b' + file[6:])

    return None


def figure_13():
    """
    Figure 13. Regions from Figure 3 complying with ε ≤ 0.3. (a) labeled
    regions. (b) tracks correspondent to (a) in their original gray
    levels. Colormaps: (a) nipy spectral. (b) magma.
    """

    image = imread(('orig_figures/dataset_01/Kr-78_4,5min/K90_incid/'
                    'K90_incid4,5min_3.bmp'), as_grey=True)

    best_arg = (10, 20)

    labels, objects, _ = ds.round_regions(image,
                                          initial_radius=best_arg[0],
                                          delta_radius=best_arg[1],
                                          toler_ecc=0.3)

    # Figure 13 (a).
    plt.figure(figsize=(15, 10))
    plt.imshow(labels, cmap='nipy_spectral')
    plt.savefig('Fig_13a.eps', bbox_inches='tight')


    # Figure 13 (b).
    plt.figure(figsize=(15, 10))
    plt.imshow(objects, cmap='magma')
    plt.savefig('Fig_13b.eps', bbox_inches='tight')

    return None


def figure_14():
    """
    Figure 14. Relation between incident energy versus mean diameter
    product ((a) 4.5 min; (c) 8.5 min samples, left Y axis) and incident
    energy versus mean gray levels ((b) 4.5 min; (d) 8.5 min samples, left
    Y axis). Cyan dashed line: electronic energy loss calculated with
    SRIM (right Y axis).
    """

    incid_energy = {'0': 865, '20': 701, '30': 613, '40': 520,
                    '50': 422, '60': 320, '70': 213, '80': 105,
                    '90': 18}

    file_45 = pd.read_csv('counting/wusem_count/roundinfo_dataset01_Kr-78_4,5min.txt')
    file_85 = pd.read_csv('counting/wusem_count/roundinfo_dataset01_Kr-78_8,5min.txt')

    kr_dedx = pd.read_csv('Kr_dEdx.txt')

    plot_color, dedx_color = ('#d95f02', '#1b9e77')

    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    data_45, data_85 = [{} for _ in range(2)]
    for idx, folder in enumerate(folders):
        data_45[samples[idx]] = file_45[file_45['folder'] == folder]
        data_85[samples[idx]] = file_85[file_85['folder'] == folder]
    data_auto45 = [data_45]
    data_auto85 = [data_85]

    # Figure 14 (a).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto45):
        for key, val in data.items():
            ax.scatter(incid_energy[key],
                       ds.px_to_um2(val['minor_axis'].mean(),
                                    val['major_axis'].mean()),
                       marker='o',
                       color=plot_color)
            ax.errorbar(incid_energy[key],
                        ds.px_to_um2(val['minor_axis'].mean(),
                                     val['major_axis'].mean()),
                        yerr=ds.px_to_um2(val['minor_axis'].std(),
                                          val['major_axis'].std()),
                        marker='o',
                        color=plot_color)

    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean diameter product ($\mu m^2$)')
    ax.invert_xaxis()

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'],
                 kr_dedx['dE/dxElec'],
                 linewidth=3,
                 linestyle='--',
                 color=dedx_color)
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_14a.eps', bbox_inches='tight')

    # Figure 14 (b).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto45):
        for key, val in data.items():
            ax.scatter(incid_energy[key],
                       val['mean_gray'].mean(),
                       marker='X',
                       color=plot_color)
            ax.errorbar(incid_energy[key],
                        val['mean_gray'].mean(),
                        yerr=val['mean_gray'].std(),
                        marker='o',
                        color=plot_color)

    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean gray shades')
    ax.invert_xaxis()

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'],
                 kr_dedx['dE/dxElec'],
                 linewidth=3,
                 linestyle='--',
                 color=dedx_color)
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_14b.eps', bbox_inches='tight')

    # Figure 14 (c).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto85):
        for key, val in data.items():
            ax.scatter(incid_energy[key],
                       ds.px_to_um2(val['minor_axis'].mean(),
                                    val['major_axis'].mean()),
                       marker='o',
                       color=plot_color)
            ax.errorbar(incid_energy[key],
                        ds.px_to_um2(val['minor_axis'].mean(),
                                     val['major_axis'].mean()),
                        yerr=ds.px_to_um2(val['minor_axis'].std(),
                                          val['major_axis'].std()),
                        marker='o',
                        color=plot_color)

    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean diameter product ($\mu m^2$)')
    ax.invert_xaxis()  # inverting X axis

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'],
                 kr_dedx['dE/dxElec'],
                 linewidth=3,
                 linestyle='--',
                 color=dedx_color)
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_14c.eps', bbox_inches='tight')

    # Figure 14 (d).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto85):
        for key, val in data.items():
            ax.scatter(incid_energy[key],
                       val['mean_gray'].mean(),
                       marker='X',
                       color=plot_color)
            ax.errorbar(incid_energy[key],
                        val['mean_gray'].mean(),
                        yerr=val['mean_gray'].std(),
                        marker='o',
                        color=plot_color)

    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean gray shades')
    ax.invert_xaxis()  # inverting X axis

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'],
                 kr_dedx['dE/dxElec'],
                 linewidth=3,
                 linestyle='--',
                 color=dedx_color)
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_14d.eps', bbox_inches='tight')

    return None


def figure_15():
    """
    Figure 15. Tracks separated in Figure 11 using the WUSEM algorithm,
    and then enumerated using the function enumerate_objects(). As in
    Figure 5, the size of the first structuring element is small when
    compared to the objects, and smaller regions where tracks overlap are
    counted as tracks. (a) Considering border tracks. (b) Ignoring border
    tracks. Parameters for WUSEM algorithm: initial_radius = 5,
    delta_radius = 2.
    """

    image = imread('figures/orig_figures/dataset_02/FT-Lab_19.07.390.MAG1.jpg',
                   as_grey=True)

    # Figure 15 (a).
    img_bin = binary_fill_holes(image < threshold_isodata(image))

    plt.figure(figsize=(15, 10))
    plt.imshow(img_bin, cmap='gray')
    plt.savefig('Fig_15a.eps', bbox_inches='tight')

    # Figure 15 (b).
    img_labels, _, _ = ds.segmentation_wusem(img_bin,
                                             initial_radius=5,
                                             delta_radius=12)
    img_labels = ds.clear_rd_border(img_labels)

    img_numbers = ds.enumerate_objects(image,
                                       img_labels,
                                       font_size=30)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_numbers, cmap='gray')
    plt.savefig('Fig_15b.eps', bbox_inches='tight')

    return None


def figure_16():
    """
    Figure 16. Comparison between manual and automatic counting for
    photomicrographs in the second dataset, when (a, b) considering border
    tracks and (c, d) ignoring border tracks. Automatic results are closer
    to manual ones when ignoring border tracks. (a, c) gray: manual
    counting. Red line: distribution median. White signal: distribution
    mean. Blue dots: outliers.

    Notes
    -----

    1. Based on the example available at:
    http://matplotlib.org/examples/pylab_examples/boxplot_demo2.html
    2. Colors extracted from the 'viridis' colormap. Code used:

    >>> from pylab import *
    >>> cmap = cm.get_cmap('viridis', 2)
    >>> for i in range(cmap.N):
    ...    rgb = cmap(i)[:3]
    ...    print(matplotlib.colors.rgb2hex(rgb))

    Code based on the example given in the best answer at:
    https://stackoverflow.com/questions/3016283/\
    create-a-color-generator-from-given-colormap-in-matplotlib
    """

    # defining some helping variables.
    manual_color, comp_color =  ('1', '0.5')
    plot_colors = {'MAG1': '#3e4989',
                   'MAG2': '#6ece58'}
    compfit_color, autofit_color = ('k', '#cb181d')
    box_colors = []

    man_count = pd.read_excel('counting/manual_count/manual_dataset02.xls')
    comp_count = pd.read_csv('counting/comp_count/comp_dataset02.txt')
    auto_count = pd.read_csv('counting/auto_count/auto_dataset02.txt')

    manual, comp, auto, mean_man = [{} for _ in range(4)]
    manual = {'MAG1': man_count.query('image <= 9'),
              'MAG2': man_count.query('image > 9')}

    # equivalent numbering for comp.
    comp = {'MAG1': comp_count.query('image <= 392'),
            'MAG2': comp_count.query('image > 392')}

    auto = {'MAG1': auto_count.query('image <= 9'),
            'MAG2': auto_count.query('image > 9')}

    # separating best candidates for each magnification.
    autobest_mag1 = auto['MAG1'][(auto['MAG1']['initial_radius'] == 5) &
                                 (auto['MAG1']['delta_radius'] == 12)]
    autobest_mag2 = auto['MAG2'][(auto['MAG2']['initial_radius'] == 10) &
                                 (auto['MAG2']['delta_radius'] == 14)]

    auto_best = {'MAG1': autobest_mag1, 'MAG2': autobest_mag2}

    man_vs_auto, box_colors = [[] for _ in range(2)]
    pos = list(range(1, 7))

    for _, val in plot_colors.items():
        box_colors.append(manual_color)
        box_colors.append(comp_color)
        box_colors.append(val)

    flier_props = dict(marker='P', markerfacecolor='#386cb0',
                       markeredgecolor='#386cb0', linestyle='none')

    x_ticks = [2, 5]
    x_labels = ['Magnification 1', 'Magnification 2']

    for key, val in manual.items():
        # data: manual, comparison, auto
        man_vs_auto.append(np.asarray(val.manual_count))
        man_vs_auto.append(np.asarray(comp[key].comp_count))
        man_vs_auto.append(np.asarray(auto_best[key].auto_count))

    # Figure 16 (a).
    fig, ax = plt.subplots(figsize=(16, 10))
    box_plot = ax.boxplot(man_vs_auto,
                          flierprops=flier_props,
                          positions=pos)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    num_boxes = len(man_vs_auto)
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
                 [np.average(man_vs_auto[i])],
                 color='w',
                 marker='P',
                 markeredgecolor='k')

    ax.set_xlabel('Sample number')
    ax.set_ylabel('Tracks counted')

    plt.savefig('Fig_16a.eps', bbox_inches='tight')

    # Figure 16 (b).
    fig, ax = plt.subplots(figsize=(16, 10))

    # preparing fit variables.
    x = np.linspace(0, 150, 1000)
    aux_manual, aux_comp, aux_auto = [[] for _ in range(3)]

    for key, val in manual.items():
        ax.plot(val.manual_count,
                auto_best[key].auto_count,
                color=plot_colors[key],
                marker='.',
                linestyle='None',
                markersize=18)

        aux_manual.append(val.manual_count.tolist())
        aux_comp.append(comp[key].comp_count.tolist())
        aux_auto.append(auto_best[key].auto_count.tolist())

    aux_manual = list(chain.from_iterable(aux_manual))
    aux_comp = list(chain.from_iterable(aux_comp))
    aux_auto = list(chain.from_iterable(aux_auto))

    # fitting a line in the data.
    fit = np.polyfit(aux_manual, aux_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(aux_manual,
            fit_fn(aux_manual),
            linewidth=3,
            color=autofit_color)
    ax.plot(x, x, '--', color='k')

    fit_comp = np.polyfit(aux_manual, aux_comp, deg=1)
    fit_fn2 = np.poly1d(fit_comp)
    ax.plot(aux_manual,
            fit_fn2(aux_manual),
            linewidth=3,
            color=compfit_color)

    # setting axes and labels.
    ax.axis([0, 120, 0, 130])
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')

    # preparing legend.
    sample = []

    for key, var in plot_colors.items():
        sample.append(mpatches.Patch(color=var,
                                     label='Magnification ' + str(key)[-1]))

    ax.legend(handles=[sample[0], sample[1]],
              loc='lower right', ncol=1, frameon=False)

    plt.savefig('Fig_16b.eps', bbox_inches='tight')

    return None


def figure_17():
    """
    Figure 17. When using suitable input parameters, WUSEM may perform
    better in certain regions where the classic watershed does not
    return reliable results. For instance, the highlighted region in (a)
    presents three tracks. WUSEM separates them correctly, but the region
    is oversegmented by the classic watershed. The highlighted region in
    (b), by its turn, is undersegmented by the classical watershed, which
    returns two tracks. WUSEM returns three tracks, being closer to the
    real number (four tracks). Left: input photomicrographs with
    highlighted regions. Center: tracks separated using WUSEM. Right:
    tracks separated using classic watershed. Parameters for WUSEM
    algorithm: initial_radius = 15, delta_radius = 4.
    """

    # Figure 17 (a), right.
    image = imread('figures/orig_figures/dataset_01/Kr-78_4,5min/K0_incid/K0_incid4,5min_2.bmp',
                   as_grey=True)
    thresh = threshold_isodata(image)
    img_bin = binary_fill_holes(image < thresh)
    img_labels, num_objects, _ = ds.segmentation_wusem(img_bin,
                                                       initial_radius=15,
                                                       delta_radius=4)
    img_number = ds.enumerate_objects(image, img_labels, font_size=25)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_number)
    plt.savefig('Fig_17a.eps', bbox_inches='tight')

    # Figure 17 (b), right.
    image = imread('figures/orig_figures/dataset_01/Kr-78_4,5min/K0_incid/K0_incid4,5min_5.bmp',
                   as_grey=True)
    thresh = threshold_isodata(image)
    img_bin = binary_fill_holes(image < thresh)
    img_labels, num_objects, _ = ds.segmentation_wusem(img_bin,
                                                       initial_radius=15,
                                                       delta_radius=4)
    img_number = ds.enumerate_objects(image, img_labels, font_size=25)

    plt.figure(figsize=(15, 10))
    plt.imshow(img_number)
    plt.savefig('Fig_17b.eps', bbox_inches='tight')

    return None


def figure_sup1():
    """
    Supplementary Figure 1: Contour map representing tracks counted in
    orig_figures/dataset_01/Kr-78_4,5min/K90_incid/K90_incid4,5min_1.bmp,
    according to the variation of initial radius and delta radius.
    The number of counted tracks decreases as initial_radius increases
    because the initial structuring element becomes larger than the ROI
    within the image. The erosion using these larger structuring elements
    removes ROI smaller than them, hence decreasing the track number.
    Increasing delta_radius decreases the number of counted tracks, but
    the difference is not significant when compared to initial radius.
    (a) Considering borders. (b) Ignoring borders. Colormap: magma.
    """

    auto = pd.read_csv('auto_count/auto_dataset01_Kr-78_4,5min_incid.txt')
    datak90_image1 = auto[(auto['folder'] == 'K90_incid') &
                          (auto['image'] == 1)]

    # initial_radius starts in 5, ends in 40 and has delta_radius 5.
    # delta_radius starts in 2, ends in 20 and has delta_radius 2.
    # let's create matrices to accomodate the countings.
    XX, YY = np.mgrid[5:41:5, 2:21:2]
    ZZk90_wb, ZZk90_nb = np.zeros(XX.shape), np.zeros(XX.shape)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        aux = int(datak90_image1.auto_count[(auto.initial_radius == i) &
                                                 (auto.delta_radius == j)])
        ZZk90_wb[(XX == i) & (YY == j)] = aux

        aux = int(datak90_image1.auto_count[(auto.initial_radius == i) &
                                               (auto.delta_radius == j)])
        ZZk90_nb[(XX == i) & (YY == j)] = aux

    # Supplementary Figure 1 (a).
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.contour(ZZk90_wb.T, colors='w')
    ax.clabel(image, fmt='%i', fontsize=default_fontsize)
    image = ax.contourf(ZZk90_wb.T, cmap='magma')

    ax.set_xlabel('initial_radius')
    ax.set_ylabel('delta_radius')
    ax.set_xticklabels(('5', '10', '15', '20', '25', '30', '35', '40'))
    ax.set_yticklabels(('2', '4', '6', '8', '10', '12', '14', '16',
                        '18', '20'))

    fig.colorbar(image, ax=ax, orientation='vertical')
    plt.savefig('Fig_sup1a.eps', bbox_inches='tight')

    # Supplementary Figure 1 (b).
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.contour(ZZk90_nb.T, colors='w')
    ax.clabel(image, fmt='%i', fontsize=default_fontsize)
    image = ax.contourf(ZZk90_nb.T, cmap='magma')

    ax.set_xlabel('initial_radius')
    ax.set_ylabel('delta_radius')
    ax.set_xticklabels(('5', '10', '15', '20', '25', '30', '35', '40'))
    ax.set_yticklabels(('2', '4', '6', '8', '10', '12', '14', '16',
                        '18', '20'))

    fig.colorbar(image, ax=ax, orientation='vertical')
    plt.savefig('Fig_sup1b.eps', bbox_inches='tight')

    return None


def figure_sup2():
    """
    Supplementary Figure 2: Analysis of the gray shade variation of each track
    from orig_figures/dataset_01/Kr-78_4,5min/K90_incid/K90_incid4,5min_1.bmp
    using contour maps. Colormap: magma.
    """

    ds.separate_tracks_set1(save_tracks=True)

    input_files = ['K90_incid4,5min_1_track_2.eps',
                   'K90_incid4,5min_1_track_4.eps',
                   'K90_incid4,5min_1_track_5.eps',
                   'K90_incid4,5min_1_track_9.eps',
                   'K90_incid4,5min_1_track_12.eps',
                   'K90_incid4,5min_1_track_13.eps',
                   'K90_incid4,5min_1_track_19.eps',
                   'K90_incid4,5min_1_track_20.eps',
                   'K90_incid4,5min_1_track_21.eps']

    # Supplementary Figure 2, from (a) to (i).
    output_files = ['Fig_sup2a.eps', 'Fig_sup2b.eps', 'Fig_sup2c.eps',
                    'Fig_sup2d.eps', 'Fig_sup2e.eps', 'Fig_sup2f.eps',
                    'Fig_sup2g.eps', 'Fig_sup2h.eps', 'Fig_sup2i.eps']

    for idx, name in enumerate(input_files):
        os.rename(src=name, dst=output_files[idx])

    return None


def figure_sup3():
    """
    Supplementary Figure 3: Input photomicrograph orig_figures/dataset_02/\
    FT-Lab_19.07.390.MAG1.jpg binarized using the ISODATA threshold
    (threshold = 0.5933) and region filling. (a) Considering border tracks.
    (b) Ignoring border tracks. Colormap: gray.
    """

    image = imread('orig_figures/dataset_02/FT-Lab_19.07.390.MAG1.jpg',
                   as_grey=True)

    imgbin_wb = binary_fill_holes(image < threshold_isodata(image))
    imgbin_nb = clear_border(binary_fill_holes(image <
                             threshold_isodata(image)))

    # Supplementary Figure 3 (a).
    plt.figure(figsize=(15, 10))
    plt.imshow(imgbin_wb, cmap='gray')
    plt.savefig('Fig_sup3a.eps', bbox_inches='tight')

    # Supplementary  Figure 3 (b).
    plt.figure(figsize=(15, 10))
    plt.imshow(imgbin_nb, cmap='gray')
    plt.savefig('Fig_sup3b.eps', bbox_inches='tight')

    return None


def figure_sup4():
    """
    Supplementary Figure 4: Contour map representing tracks counted
    according to the variation of initial_radius and delta_radius for
    tracks in orig_figures/dataset_02/FT-Lab_19.07.390.MAG1.jpg. The number
    of counted tracks also decreases as initial_radius increases. Increasing
    delta_radius makes no difference in the number of counted tracks when
    initial_radius is higher than 10. (a) Considering borders. (b) Ignoring
    borders. Colormap: magma.
    """

    auto_set2 = pd.read_csv('auto_count/auto_dataset02.txt')
    auto = {'MAG1': auto_set2.query('image <= 8 or image == 19'),
            'MAG2': auto_set2.query('image > 8 and image < 18')}
    image = auto['MAG1'][auto['MAG1']['image'] == 6]

    # initial_radius starts in 5, ends in 40 and has delta_radius 5.
    # delta_radius starts in 2, ends in 20 and has delta_radius 2.
    # let's create matrices to accomodate the countings.
    XX, YY = np.mgrid[5:41:5, 2:21:2]
    ZZ_wb, ZZ_nb = np.zeros(XX.shape), np.zeros(XX.shape)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        aux = int(image.auto_count[(auto_set2.initial_radius == i) &
                                        (auto_set2.delta_radius == j)])
        ZZ_wb[(XX == i) & (YY == j)] = aux

        aux = int(image.auto_count[(auto_set2.initial_radius == i) &
                                      (auto_set2.delta_radius == j)])
        ZZ_nb[(XX == i) & (YY == j)] = aux

    # Supplementary Figure 4 (a).
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.contour(ZZ_wb.T, colors='w')
    ax.clabel(image, fmt='%i', fontsize=default_fontsize)
    image = ax.contourf(ZZ_wb.T, cmap='magma')

    ax.set_xlabel('initial_radius')
    ax.set_ylabel('delta_radius')
    ax.set_xticklabels(('5', '10', '15', '20', '25', '30', '35', '40'))
    ax.set_yticklabels(('2', '4', '6', '8', '10', '12', '14', '16', '18',
                        '20'))

    fig.colorbar(image, ax=ax, orientation='vertical')
    plt.savefig('Fig_sup4a.eps', bbox_inches='tight')

    # Supplementary Figure 4 (b).
    fig, ax = plt.subplots(figsize=(12, 12))
    image = ax.contour(ZZ_nb.T, colors='w')
    ax.clabel(image, fmt='%i', fontsize=default_fontsize)
    image = ax.contourf(ZZ_nb.T, cmap='magma')

    ax.set_xlabel('initial_radius')
    ax.set_ylabel('delta_radius')
    ax.set_xticklabels(('5', '10', '15', '20', '25', '30', '35', '40'))
    ax.set_yticklabels(('2', '4', '6', '8', '10', '12', '14', '16', '18',
                        '20'))

    fig.colorbar(image, ax=ax, orientation='vertical')
    plt.savefig('Fig_sup4b.eps', bbox_inches='tight')

    return None


def figure_sup5():
    """
    Supplementary Figure 5: Manual counting mean (top of the blue bar;
    values on the right) for each sample and automatic counting results
    with mean within (yellow points) and outside (gray points) the
    tolerance interval (blue bar) for the second dataset. (a) Considering
    borders. (b) Ignoring borders.
    """

    # defining mean tolerance used in the paper.
    tol = 10
    man_color = '#386cb0'
    auto_color = '#fdc086'
    plot_where = {'MAG1': 10, 'MAG2': 20}
    ticks = [10, 20]

    man_count = pd.read_excel('manual_count/manual_dataset02.xls')
    auto_count = pd.read_csv('auto_count/auto_dataset02.txt')

    manual, auto, meanman_wb, meanman_nb = [{} for _ in range(4)]
    manual = {'MAG1': man_count.query('image <= 8 or image == 19'),
              'MAG2': man_count.query('image > 8 and image < 18')}

    auto = {'MAG1': auto_count.query('image <= 8 or image == 19'),
            'MAG2': auto_count.query('image > 8 and image < 18')}

    # calculating the means for manual counting, and obtaining the
    # best candidates for initial_radius and delta_radius.

    for key, val in manual.items():
        # with border.
        meanman_wb[key] = val.manual_count.mean()
        # without border.
        meanman_nb[key] = val.manual_count.mean()

    # Supplementary Figure 5 (a).
    fig, ax = plt.subplots(figsize=(15, 10))

    for key, val in meanman_wb.items():
        ax.fill_between(np.arange(plot_where[key]-0.5,
                                  plot_where[key]+0.8),
                        y1=val, y2=val-tol,
                        facecolor=man_color)
        ax.annotate(np.round(val, decimals=1),
                    xy=(plot_where[key]+1.2, val), color=man_color)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        for key, val in auto.items():
            aux_wb = val.auto_count[(val.initial_radius == i) &
                                         (val.delta_radius == j)].mean()
            if 0 < (meanman_wb[key] - aux_wb) < tol:
                ax.scatter(plot_where[key], aux_wb, color=auto_color,
                           edgecolor='k')
            else:
                ax.scatter(plot_where[key], aux_wb, color='0.8',
                           edgecolor='k')

    ax.set_xticklabels(('Magnification 1', 'Magnification 2'))
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean counting')
    ax.set_xticks(ticks)
    ax.set_xlim([5, 25])
    ax.set_ylim([-1, 100])

    plt.savefig('Fig_sup5a.eps', bbox_inches='tight')

    # Supplementary Figure 5 (b).
    fig, ax = plt.subplots(figsize=(15, 10))

    for key, val in meanman_nb.items():
        ax.fill_between(np.arange(plot_where[key]-0.5,
                                  plot_where[key]+0.8),
                        y1=val, y2=val-tol,
                        facecolor=man_color)
        ax.annotate(np.round(val, decimals=1),
                    xy=(plot_where[key]+1.2, val), color=man_color)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        for key, val in auto.items():
            aux_nb = val.auto_count[(val.initial_radius == i) &
                                       (val.delta_radius == j)].mean()
            if 0 < (meanman_nb[key] - aux_nb) < tol:
                ax.scatter(plot_where[key], aux_nb, color=auto_color,
                           edgecolor='k')
            else:
                ax.scatter(plot_where[key], aux_nb, color='0.8',
                           edgecolor='k')

    ax.set_xticklabels(('Magnification 1', 'Magnification 2'))
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean counting')
    ax.set_xticks(ticks)
    ax.set_xlim([5, 25])
    ax.set_ylim([-1, 100])

    plt.savefig('Fig_sup5b.eps', bbox_inches='tight')

    return None


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


def _aux_figs_10_11_(man_count, wusem_count, water_count, hwater_count,
                    radii_best, seed_best):

    # defining some helping variables.
    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    pos = list(range(1, 5))  # manual, watershed, h-watershed and wusem

    manual_color = '#ffffff'
    water_color = '#d3d3d3'
    hwater_color = '#778899'
    wusem_colors = ['#440154', '#482878', '#3e4989',
                    '#31688e', '#26828e', '#35b779',
                    '#6ece58', '#b5de2b', '#fde725']

    box_colors = []    
    for color in wusem_colors:
        box_colors.append([manual_color, water_color, hwater_color,
                           color])


    flier_props = dict(marker='P', markerfacecolor='#386cb0',
                       markeredgecolor='#386cb0', linestyle='none')

    x_labels = ['K0', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70',
                'K80', 'K90']

    manual, wusem, wusem_best, water, hwater, hwater_best = [{} for _ in range(6)]

    for idx, folder in enumerate(folders):
        manual[samples[idx]] = man_count[man_count['folder'] == folder]
        wusem[samples[idx]] = wusem_count[wusem_count['folder'] == folder]
        water[samples[idx]] = water_count[water_count['folder'] == folder]
        hwater[samples[idx]] = hwater_count[hwater_count['folder'] == folder]

    # best candidate for WUSEM.
    for key, val in wusem.items():
        wusem_best[key] = val[(val['initial_radius'] == radii_best[0]) &
                              (val['delta_radius'] == radii_best[1])]

    # best candidate for H-watershed.
    for key, val in hwater.items():
        hwater_best[key] = val[val['seed'] == seed_best]

    man_vs_auto = []

    for key, val in manual.items():
        # data: manual, classic watershed, h-watershed, wusem
        man_vs_auto.append([np.asarray(val.manual_count),
                            np.asarray(water[key].auto_count),
                            np.asarray(hwater_best[key].auto_count),
                            np.asarray(wusem_best[key].auto_count)])

    for idx, comp in enumerate(man_vs_auto):
        fig, ax = plt.subplots(figsize=(10, 10))
        box_plot = ax.boxplot(comp,
                              flierprops=flier_props,
                              positions=pos)

        ax.xaxis.set_major_locator(plt.NullLocator())  # deleting ticks.
        ax.set_xlabel(x_labels[idx])

        num_boxes = len(comp)
        medians = list(range(num_boxes))

        for i in range(num_boxes):
            box = box_plot['boxes'][i]
            boxX, boxY = [[] for _ in range(2)]
            for j in range(5):
                boxX.append(box.get_xdata()[j])
                boxY.append(box.get_ydata()[j])
            box_coords = list(zip(boxX, boxY))
            box_polygon = mpatches.Polygon(box_coords,
                                           facecolor=box_colors[idx][i])
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
                     [np.average(man_vs_auto[i])],
                     color='w', marker='P', markeredgecolor='k')

        if x_labels[idx] in ['K0', 'K40', 'K70']:
            ax.set_ylabel('Tracks counted')

        plt.savefig('Fig-' + x_labels[idx] + '.eps',
                    bbox_inches='tight')

    return None


def _aux_fig_12_(man_count, wusem_count, water_count, hwater_count,
                 radii_best, seed_best):


    # defining some helping variables.
    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    manual_color = '1'
    water_color = '0.75'
    hwater_color = '0.5'
    wusem_colors = ['#440154', '#482878', '#3e4989',
                    '#31688e', '#26828e', '#35b779',
                    '#6ece58', '#b5de2b', '#fde725']

    manual, wusem, wusem_best, water, hwater, hwater_best = [{} for _ in range(6)]

    for idx, folder in enumerate(folders):
        manual[samples[idx]] = man_count[man_count['folder'] == folder]
        wusem[samples[idx]] = wusem_count[wusem_count['folder'] == folder]
        water[samples[idx]] = water_count[water_count['folder'] == folder]
        hwater[samples[idx]] = hwater_count[hwater_count['folder'] == folder]

    # best candidate for WUSEM.
    for key, val in wusem.items():
        wusem_best[key] = val[(val['initial_radius'] == radii_best[0]) &
                              (val['delta_radius'] == radii_best[1])]

    # best candidate for H-watershed.
    for key, val in hwater.items():
        hwater_best[key] = val[val['seed'] == seed_best]

    fig, ax = plt.subplots(figsize=(16, 10))

    # preparing fit variables.
    x = np.linspace(0, 50, 1000)
    aux_manual, aux_water, aux_hwater, aux_wusem = [[] for _ in range(4)]

    for idx, (key, val) in enumerate(manual.items()):
        ax.plot(val.manual_count,
                wusem_best[key].auto_count,
                color=wusem_colors[idx],
                marker='.',
                linestyle='None',
                markersize=18)

        aux_manual.append(val.manual_count.tolist())
        aux_water.append(water[key].auto_count.tolist())
        aux_hwater.append(hwater_best[key].auto_count.tolist())
        aux_wusem.append(wusem_best[key].auto_count.tolist())

    aux_manual = list(chain.from_iterable(aux_manual))
    aux_water = list(chain.from_iterable(aux_water))
    aux_hwater = list(chain.from_iterable(aux_hwater))
    aux_wusem = list(chain.from_iterable(aux_wusem))

    # fitting a line in the data.
    # wusem
    fit = np.polyfit(aux_manual, aux_wusem, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(aux_manual, fit_fn(aux_manual), linewidth=4, color=wusem_colors[4])
    ax.plot(x, x, '--', color='k')

    # classic watershed
    fit_water = np.polyfit(aux_manual, aux_water, deg=1)
    fit_fn2 = np.poly1d(fit_water)
    ax.plot(aux_manual, fit_fn2(aux_manual), linewidth=4, color=water_color)

    # h-watershed
    fit_hwater = np.polyfit(aux_manual, aux_hwater, deg=1)
    fit_fn3 = np.poly1d(fit_hwater)
    ax.plot(aux_manual, fit_fn3(aux_manual), linewidth=4, color=hwater_color)

    # setting axes and labels.
    ax.axis([0, 50, 0, 50])
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('WUSEM counting')

    # preparing legend.
    sample = []

    for idx, key in enumerate(manual.keys()):
        sample.append(mpatches.Patch(color=wusem_colors[idx],
                                     label='Sample K' + str(key)))

    ax.legend(handles=[sample[0], sample[1], sample[2],
                       sample[3], sample[4], sample[5],
                       sample[6], sample[7], sample[8]],
              fontsize=20, loc='lower right', ncol=1, frameon=False)

    plt.savefig('Fig_12.eps', bbox_inches='tight')

    return None
