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

from itertools import chain, product
from matplotlib import mlab
from scipy.ndimage.morphology import (binary_fill_holes,
                                      distance_transform_edt)
from scipy.stats import norm
from skimage.filters import threshold_isodata
from skimage.io import imread
from skimage.measure import label
from skimage.morphology import binary_erosion, disk, watershed
from skimage.segmentation import clear_border

import desiqueira2017 as ds
from matplotlib.animation import ArtistAnimation
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import os
import pandas as pd
import warnings

# Setting up the figures appearance.
plt.rcParams['font.family'] = 'monospace'
plt.rcParams['font.size'] = 20
plt.rcParams['axes.labelsize'] = plt.rcParams['font.size']
plt.rcParams['axes.titlesize'] = 1.2*plt.rcParams['font.size']
plt.rcParams['legend.fontsize'] = plt.rcParams['font.size']
plt.rcParams['xtick.labelsize'] = plt.rcParams['font.size']
plt.rcParams['ytick.labelsize'] = plt.rcParams['font.size']

default_fontsize = 15

# Ignoring warnings.
warnings.filterwarnings('ignore')


def figure_1():
    """
    Figure 1: WUSEM algorithm application in an input image. First, the
    image is binarized using the ISODATA threshold. Then, the image is
    eroded using initial radius (r0) and iterative radius (∆r) equal to
    1 and 1 respectively, to ease visualization. The process continues
    until the eroded image has regions in it. All erosions are summed,
    and the result is labeled; each labeled track receives a color from
    the nipy_spectral colormap. Finally, the function enumerate_objects()
    is used to number the found tracks. Final results are shown to r0
    and ∆r equal to 25 and 2, respectively. Animation also available at
    https://youtu.be/gYKbqMEOhB0.
    """

    image_animation = []

    img_orig = imread(('orig_figures/dataset_01/Kr-78_4,5min/K0_incid/'
                       'K0_incid4,5min_1.bmp'), as_grey=True)

    fig, ax = plt.subplots(figsize=(15, 10), ncols=1, nrows=1,
                           tight_layout=True)
    ax.set_aspect('equal')

    # 1st image: original photomicrograph.
    curr_frame = ax.imshow(img_orig, cmap='gray')
    for i in range(10):
        image_animation.append([curr_frame])

    # 2nd image: binary image.
    aux = img_orig < threshold_isodata(img_orig)
    image = clear_border(binary_fill_holes(aux))
    curr_frame = ax.imshow(image, cmap='gray')
    for i in range(10):
        image_animation.append([curr_frame])

    rows, cols = image.shape
    distance = distance_transform_edt(image)

    # following images: erosions.
    initial_radius, delta_radius = 1, 1
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius

    while True:
        erod_aux = binary_erosion(image, selem=disk(curr_radius))
        curr_frame = ax.imshow(erod_aux, cmap='gray')
        image_animation.append([curr_frame])

        if erod_aux.min() == erod_aux.max():
            break

        markers = label(erod_aux)
        curr_labels = watershed(-distance,
                                markers,
                                mask=image)
        img_labels += curr_labels
        curr_radius += delta_radius

    # following image: chosen segmentation.
    initial_radius, delta_radius = 25, 2
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
    img_labels = label(img_labels)
    curr_frame = ax.imshow(img_labels, cmap='nipy_spectral')
    for i in range(10):
        image_animation.append([curr_frame])

    # last image: numbered image.
    img_number = ds.enumerate_objects(img_orig,
                                      img_labels,
                                      font_size=30)

    curr_frame = ax.imshow(img_number)
    for i in range(10):
        image_animation.append([curr_frame])

    # Figure 1.
    ani = ArtistAnimation(fig, image_animation, interval=350, blit=True)
    ani.save('Fig_1.mp4', bitrate=-1, codec='libx264')

    return None


def figure_4():
    """
    Figure 4: Input photomicrograph binarized using the ISODATA threshold
    (threshold = 128) and region filling. (a) Considering border tracks.
    (b) Ignoring border tracks. Colormap: gray. Note that tracks connected
    to the border tracks are also removed by the algorithm, while they
    would be counted by the observer.
    """

    image = imread(('orig_figures/dataset_01/Kr-78_4,5min/K90_incid/'
                    'K90_incid4,5min_1.bmp'), as_grey=True)

    imgbin_wb = binary_fill_holes(image < threshold_isodata(image))
    imgbin_nb = clear_border(binary_fill_holes(image <
                             threshold_isodata(image)))

    # Figure 4 (a).
    plt.figure(figsize=(10, 12))
    plt.imshow(imgbin_wb, cmap='gray')
    plt.savefig('Fig_4a.eps', bbox_inches='tight')

    # Figure 4 (b).
    plt.figure(figsize=(10, 12))
    plt.imshow(imgbin_nb, cmap='gray')
    plt.savefig('Fig_4b.eps', bbox_inches='tight')

    return None


def figure_5():
    """
    Figure 5: Tracks separated in Figure 3 using the WUSEM algorithm and
    enumerated using enumerate objects(). Since the size of the first
    structuring element is small when compared to the objects, smaller
    regions where tracks overlap are counted as tracks, e.g.: (a) 7, 8, 9,
    and 11 (only two tracks). (a) Considering border tracks. (b) Ignoring
    border tracks. Parameters for WUSEM algorithm: initial_radius = 5,
    delta_radius = 2.
    """

    image = imread(('orig_figures/dataset_01/Kr-78_4,5min/K90_incid/'
                    'K90_incid4,5min_1.bmp'), as_grey=True)

    # Figure 5 (a).
    imgbin_wb = binary_fill_holes(image < threshold_isodata(image))
    imglabel_wb, _, _ = ds.segmentation_wusem(imgbin_wb, initial_radius=5,
                                              delta_radius=2)
    imgnumber_wb = ds.enumerate_objects(image, imglabel_wb, font_size=25)

    plt.figure(figsize=(10, 12))
    plt.imshow(imgnumber_wb, cmap='gray')
    plt.savefig('Fig_5a.eps', bbox_inches='tight')

    # Figure 5 (b).
    imgbin_nb = clear_border(binary_fill_holes(image <
                             threshold_isodata(image)))
    imglabel_nb, _, _ = ds.segmentation_wusem(imgbin_nb, initial_radius=5,
                                              delta_radius=2)
    imgnumber_nb = ds.enumerate_objects(image, imglabel_nb, font_size=25)

    plt.figure(figsize=(10, 12))
    plt.imshow(imgnumber_nb, cmap='gray')
    plt.savefig('Fig_5b.eps', bbox_inches='tight')

    return None


def figure_6():
    """
    Figure 6: Manual counting mean (top of the blue bar; values on the
    right) for each sample and automatic counting results with mean within
    (orange points) and outside (gray points) the tolerance interval (blue
    bar) for the second dataset. (a) Considering borders. (b) Ignoring
    borders.
    """

    # defining some helping variables.
    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    # defining mean tolerance used in the paper.
    tol = 2.5
    man_color = '#386cb0'
    auto_color = '#fdc086'
    plot_where = {'0': 10, '20': 20, '30': 30, '40': 40, '50': 50,
                  '60': 60, '70': 70, '80': 80, '90': 90}
    ticks = [10, 20, 30, 40, 50, 60, 70, 80, 90]

    man_count = pd.read_excel('manual_count/manual_Kr-78_4,5min.xls')
    auto_count = pd.read_csv('auto_count/autoincid_Kr-78_4,5min.txt')
    manual, auto = {}, {}

    for idx, folder in enumerate(folders):
        manual[samples[idx]] = man_count[man_count['folder'] == folder]
        auto[samples[idx]] = auto_count[auto_count['folder'] == folder]

    # calculating the means for manual counting, and obtaining the
    # best candidates for initial_radius and delta_radius.
    meanman_wb, meanman_nb = [{} for _ in range(2)]

    for key, val in manual.items():
        # with border.
        meanman_wb[key] = val.manual_withborder.mean()
        # without border.
        meanman_nb[key] = val.manual_noborder.mean()

    # Figure 6 (a).
    fig, ax = plt.subplots(figsize=(15, 10))

    for key, val in meanman_wb.items():
        ax.fill_between(np.arange(plot_where[key]-1,
                                  plot_where[key]+2),
                        y1=val, y2=val-tol,
                        facecolor=man_color)
        ax.annotate(np.round(val, decimals=1),
                    xy=(plot_where[key]+1.2, val), color=man_color)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        for key, val in auto.items():
            aux_wb = val.auto_withborder[(val.initial_radius == i) &
                                         (val.delta_radius == j)].mean()
            if 0 < (meanman_wb[key] - aux_wb) < tol:
                ax.scatter(plot_where[key], aux_wb, color=auto_color,
                           edgecolor='k')
            else:
                ax.scatter(plot_where[key], aux_wb, color='0.8',
                           edgecolor='k')

    ax.set_xticklabels(('K0', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70',
                        'K80', 'K90'))
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean counting')
    ax.set_xticks(ticks)
    ax.set_xlim([5, 100])
    ax.set_ylim([-1, 40])

    plt.savefig('Fig_6a.eps', bbox_inches='tight')

    # Figure 6 (b).
    fig, ax = plt.subplots(figsize=(15, 10))

    for key, val in meanman_nb.items():
        ax.fill_between(np.arange(plot_where[key]-1,
                                  plot_where[key]+2),
                        y1=val, y2=val-tol,
                        facecolor=man_color)
        ax.annotate(np.round(val, decimals=1),
                    xy=(plot_where[key]+1.2, val), color=man_color)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        for key, val in auto.items():
            aux_nb = val.auto_noborder[(val.initial_radius == i) &
                                       (val.delta_radius == j)].mean()
            if 0 < (meanman_nb[key] - aux_nb) < tol:
                ax.scatter(plot_where[key], aux_nb, color=auto_color,
                           edgecolor='k')
            else:
                ax.scatter(plot_where[key], aux_nb, color='0.8',
                           edgecolor='k')

    ax.set_xticklabels(('K0', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70',
                        'K80', 'K90'))
    ax.set_xlabel('Samples')
    ax.set_ylabel('Mean counting')
    ax.set_xticks(ticks)
    ax.set_xlim([5, 100])
    ax.set_ylim([-1, 40])

    plt.savefig('Fig_6b.eps', bbox_inches='tight')

    return None


def figure_7():
    """
    Figure 7: Comparison between manual and automatic counting for 4.5
    min etching samples, when (a, b) considering border tracks and (c, d)
    ignoring border tracks. Compared to manual counting, automatic results
    variate less when ignoring borders. (a, c) gray: manual counting. Red
    line: distribution median. White signal: distribution mean. (b, d)
    dashed: 1:1 line. Red line: regression for the experimental data.

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

    # defining some helping variables.
    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    man_count = pd.read_excel('manual_count/manual_Kr-78_4,5min.xls')
    auto_count = pd.read_csv('auto_count/autoincid_Kr-78_4,5min.txt')
    manual, auto, autobest_wb, autobest_nb = [{} for _ in range(4)]

    for idx, folder in enumerate(folders):
        manual[samples[idx]] = man_count[man_count['folder'] == folder]
        auto[samples[idx]] = auto_count[auto_count['folder'] == folder]

    for key, val in auto.items():
        # best candidate for "with borders" scenario.
        autobest_wb[key] = val[(val['initial_radius'] == 5) &
                               (val['delta_radius'] == 4)]
        # best candidate for "no borders" scenario.
        autobest_nb[key] = val[(val['initial_radius'] == 25) &
                               (val['delta_radius'] == 2)]

    manvsauto_wb, manvsauto_nb = [[] for _ in range(2)]
    pos = list(range(1, 19))

    plot_colors = {'0': '#440154', '20': '#482878', '30': '#3e4989',
                   '40': '#31688e', '50': '#26828e', '60': '#35b779',
                   '70': '#6ece58', '80': '#b5de2b', '90': '#fde725'}
    color_fit = '#e41a1c'
    box_colors = []

    for _, val in plot_colors.items():
        box_colors.append('0.80')
        box_colors.append(val)

    flier_props = dict(marker='P', markerfacecolor='#386cb0',
                       markeredgecolor='#386cb0', linestyle='none')

    x_ticks = [1.5, 3.5, 5.5, 7.5, 9.5, 11.5, 13.5, 15.5, 17.5]
    x_labels = ['K0', 'K20', 'K30', 'K40', 'K50', 'K60', 'K70',
                'K80', 'K90']

    for key, val in manual.items():
        # data for "with borders" scenario.
        manvsauto_wb.append([np.asarray(val.manual_withborder)])
        manvsauto_wb.append([np.asarray(autobest_wb[key].auto_withborder)])
        # data for "no borders" scenario.
        manvsauto_nb.append([np.asarray(val.manual_noborder)])
        manvsauto_nb.append([np.asarray(autobest_nb[key].auto_noborder)])

    # Figure 7 (a).
    fig, ax = plt.subplots(figsize=(12, 12))
    box_plot = ax.boxplot(manvsauto_wb, flierprops=flier_props,
                          positions=pos)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    num_boxes = len(manvsauto_wb)
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
                 [np.average(manvsauto_wb[i])],
                 color='w', marker='P', markeredgecolor='k')

    ax.set_xlabel('Sample number')
    ax.set_ylabel('Tracks counted')

    plt.savefig('Fig_7a.eps', bbox_inches='tight')

    # Figure 7 (b).
    fig, ax = plt.subplots(figsize=(12, 12))

    # preparing fit variables.
    x = np.linspace(0, 50, 1000)
    aux_manual, aux_auto = [[] for _ in range(2)]

    for key, val in manual.items():
        ax.plot(val.manual_withborder,
                autobest_wb[key].auto_withborder,
                color=plot_colors[key],
                marker='.',
                linestyle='None',
                markersize=18)

        aux_manual.append(val.manual_withborder.tolist())
        aux_auto.append(autobest_wb[key].auto_withborder.tolist())

    aux_manual = list(chain.from_iterable(aux_manual))
    aux_auto = list(chain.from_iterable(aux_auto))

    # fitting a line in the data.
    fit = np.polyfit(aux_manual, aux_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(aux_manual, fit_fn(aux_manual), linewidth=3, color=color_fit)
    ax.plot(x, x, '--', color='k')

    # setting axes and labels.
    ax.axis([0, 50, 0, 50])
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')

    # preparing legend.
    sample = []

    for key, var in plot_colors.items():
        sample.append(mpatches.Patch(color=var,
                                     label='Sample K' + str(key)))

    ax.legend(handles=[sample[0], sample[1], sample[2],
                       sample[3], sample[4], sample[5],
                       sample[6], sample[7], sample[8]],
              loc='lower right', ncol=1, frameon=False)

    plt.savefig('Fig_7b.eps', bbox_inches='tight')

    # Figure 7 (c).
    fig, ax = plt.subplots(figsize=(12, 12))
    box_plot = ax.boxplot(manvsauto_nb, flierprops=flier_props,
                          positions=pos)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    num_boxes = len(manvsauto_nb)
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
                 [np.average(manvsauto_nb[i])],
                 color='w', marker='P', markeredgecolor='k')

    ax.set_xlabel('Sample number')
    ax.set_ylabel('Tracks counted')

    plt.savefig('Fig_7c.eps', bbox_inches='tight')

    # Figure 7 (d).
    fig, ax = plt.subplots(figsize=(12, 12))

    # preparing fit variables.
    x = np.linspace(0, 50, 1000)
    aux_manual, aux_auto = [[] for _ in range(2)]

    for key, val in manual.items():
        ax.plot(val.manual_noborder,
                autobest_wb[key].auto_noborder,
                color=plot_colors[key],
                marker='.',
                linestyle='None',
                markersize=18)

        aux_manual.append(val.manual_noborder.tolist())
        aux_auto.append(autobest_nb[key].auto_noborder.tolist())

    aux_manual = list(chain.from_iterable(aux_manual))
    aux_auto = list(chain.from_iterable(aux_auto))

    # fitting a line in the data.
    fit = np.polyfit(aux_manual, aux_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(aux_manual, fit_fn(aux_manual), linewidth=3, color=color_fit)
    ax.plot(x, x, '--', color='k')

    # setting axes and labels.
    ax.axis([0, 50, 0, 50])
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')

    # preparing legend.
    sample = []

    for key, var in plot_colors.items():
        sample.append(mpatches.Patch(color=var,
                                     label='Sample K' + str(key)))

    ax.legend(handles=[sample[0], sample[1], sample[2],
                       sample[3], sample[4], sample[5],
                       sample[6], sample[7], sample[8]],
              loc='lower right', ncol=1, frameon=False)

    plt.savefig('Fig_7d.eps', bbox_inches='tight')

    return None


def figure_8():
    """
    Figure 8: Histogram of minor diameters (D<) of regions in the samples
    4.5 min as separated by WUSEM algorithm, considering borders scenario.
    WUSEM seems to introduce small regions in the images, as seen in (a);
    we use only regions within the interval μ ± 3σ to have reliable results.
    (a) histogram of all samples (μ = 62.887, σ = 13.329). (b) histogram
    of samples within μ ± 3σ (μ = 65.555, σ = 2.9062). Dashed lines:
    normal probability density functions (PDF) fitted.
    """

    # Figure 8 (a).
    info_tracks = pd.read_csv('auto_count/roundinfo_Kr-78_4,5min_T.txt')
    fig, ax = plt.subplots(figsize=(12, 8))

    (mu, sigma) = norm.fit(info_tracks['minor_axis'])
    n, bins, patches = ax.hist(info_tracks['minor_axis'], bins=25, normed=True,
                               color='#386cb0', edgecolor='k')
    fit = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, fit, 'k--', linewidth=2)
    ax.set_xlabel('Minor diameter ($D_{<}$, $\mu m^{2}$)')
    ax.set_ylabel('Normalized frequency')

    plt.savefig('Fig_8a.eps', bbox_inches='tight')

    # Figure 8 (b).
    info_clean = pd.read_csv('auto_count/roundclean_Kr-78_4,5min_T.txt')
    fig, ax = plt.subplots(figsize=(12, 8))

    (mu, sigma) = norm.fit(info_clean['minor_axis'])
    n, bins, patches = ax.hist(info_clean['minor_axis'], bins=25, normed=True,
                               color='#386cb0', edgecolor='k')
    fit = mlab.normpdf(bins, mu, sigma)
    ax.plot(bins, fit, 'k--', linewidth=2)
    ax.set_xlabel('Minor diameter ($D_{<}$, $\mu m^{2}$)')
    ax.set_ylabel('Normalized frequency')

    plt.savefig('Fig_8b.eps', bbox_inches='tight')

    return None


def figure_9():
    """
    Figure 9: Regions from Figure 3 complying with ε ≤ 0.3. (a) labeled
    regions. (b) tracks correspondent to (a) in their original gray levels.
    Colormaps: (a) nipy spectral. (b) magma.
    """

    image = imread(('orig_figures/dataset_01/Kr-78_4,5min/K90_incid/'
                    'K90_incid4,5min_1.bmp'), as_grey=True)
    best_arg = (5, 4)

    labels, objects, _ = ds.round_regions(image, initial_radius=best_arg[0],
                                          delta_radius=best_arg[1],
                                          toler_ecc=0.3, count_border=True)

    # Figure 9 (a).
    plt.figure(figsize=(15, 10))
    plt.imshow(labels, cmap='nipy_spectral')
    plt.savefig('Fig_9a.eps', bbox_inches='tight')

    # Figure 9 (b).
    plt.figure(figsize=(15, 10))
    plt.imshow(objects, cmap='magma')
    plt.savefig('Fig_9b.eps', bbox_inches='tight')

    return None


def figure_10():
    """
    Figure 10: Relation between incident energy versus mean diameter
    product ((a) 4.5 min; (c) 5.5 min samples, left Y axis) and incident
    energy versus mean gray levels ((b) 4.5 min; (d) 8.5 min samples, left
    Y axis). Purple dashed line: electronic energy loss calculated with
    SRIM (right Y axis).
    """

    incid_energy = {'0': 865, '20': 701, '30': 613, '40': 520,
                    '50': 422, '60': 320, '70': 213, '80': 105,
                    '90': 18}

    file_wb45 = pd.read_csv('auto_count/roundclean_Kr-78_4,5min_T.txt')
    file_nb45 = pd.read_csv('auto_count/roundclean_Kr-78_4,5min_F.txt')
    file_wb85 = pd.read_csv('auto_count/roundclean_Kr-78_8,5min_T.txt')
    file_nb85 = pd.read_csv('auto_count/roundclean_Kr-78_8,5min_F.txt')

    kr_dedx = pd.read_csv('Kr_dEdx.txt')

    plot_colors = ['#1b9e77', '#d95f02']
    samples = ['0', '20', '30', '40', '50', '60', '70', '80', '90']
    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    # preparing legend.
    border = []

    for color in plot_colors:
        border.append(mpatches.Patch(color=color))

    data_wb45, data_nb45 = [{} for _ in range(2)]
    for idx, folder in enumerate(folders):
        data_wb45[samples[idx]] = file_wb45[file_wb45['folder'] == folder]
        data_nb45[samples[idx]] = file_nb45[file_nb45['folder'] == folder]
    data_auto45 = [data_wb45, data_nb45]

    # Figure 10 (a).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto45):
        for key, val in data.items():
            ax.scatter(incid_energy[key],
                       ds.px_to_um2(val['minor_axis'].mean(),
                                    val['major_axis'].mean()),
                       marker='o', color=plot_colors[idx])
            ax.errorbar(incid_energy[key],
                        ds.px_to_um2(val['minor_axis'].mean(),
                                     val['major_axis'].mean()),
                        yerr=ds.px_to_um2(val['minor_axis'].std(),
                                          val['major_axis'].std()),
                        marker='o', color=plot_colors[idx])

    ax.legend(handles=[border[0], border[1]],
              labels=['Considering borders', 'Ignoring borders'],
              loc='upper left', ncol=1, frameon=False)
    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean diameter product ($\mu m^2$)')
    ax.invert_xaxis()

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'], kr_dedx['dE/dxElec'],
                 linewidth=3, linestyle='--', color='#7570b3')
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_10a.eps', bbox_inches='tight')

    # Figure 10 (b).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto45):
        for key, val in data.items():
            ax.scatter(incid_energy[key], val['mean_gray'].mean(),
                       marker='X', color=plot_colors[idx])
            ax.errorbar(incid_energy[key], val['mean_gray'].mean(),
                        yerr=val['mean_gray'].std(), marker='o',
                        color=plot_colors[idx])

    ax.legend(handles=[border[0], border[1]],
              labels=['Considering borders', 'Ignoring borders'],
              loc='upper left', ncol=1, frameon=False)
    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean gray shades')
    ax.invert_xaxis()

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'], kr_dedx['dE/dxElec'],
                 linewidth=3, linestyle='--', color='#7570b3')
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_10b.eps', bbox_inches='tight')

    data_wb85, data_nb85 = [{} for _ in range(2)]
    for idx, folder in enumerate(folders):
        data_wb85[samples[idx]] = file_wb85[file_wb85['folder'] == folder]
        data_nb85[samples[idx]] = file_nb85[file_nb85['folder'] == folder]
    data_auto85 = [data_wb85, data_nb85]

    # Figure 10 (c).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto85):
        for key, val in data.items():
            ax.scatter(incid_energy[key],
                       ds.px_to_um2(val['minor_axis'].mean(),
                                    val['major_axis'].mean()),
                       marker='o', color=plot_colors[idx])
            ax.errorbar(incid_energy[key],
                        ds.px_to_um2(val['minor_axis'].mean(),
                                     val['major_axis'].mean()),
                        yerr=ds.px_to_um2(val['minor_axis'].std(),
                                          val['major_axis'].std()),
                        marker='o', color=plot_colors[idx])

    ax.legend(handles=[border[0], border[1]],
              labels=['Considering borders', 'Ignoring borders'],
              loc='upper left', ncol=1, frameon=False)
    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean diameter product ($\mu m^2$)')
    ax.invert_xaxis()  # inverting X axis

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'], kr_dedx['dE/dxElec'],
                 linewidth=3, linestyle='--', color='#7570b3')
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_10c.eps', bbox_inches='tight')

    # Figure 10 (d).
    fig, ax = plt.subplots(figsize=(15, 10))

    for idx, data in enumerate(data_auto85):
        for key, val in data.items():
            ax.scatter(incid_energy[key], val['mean_gray'].mean(),
                       marker='X', color=plot_colors[idx])
            ax.errorbar(incid_energy[key], val['mean_gray'].mean(),
                        yerr=val['mean_gray'].std(), marker='o',
                        color=plot_colors[idx])

    ax.legend(handles=[border[0], border[1]],
              labels=['Considering borders', 'Ignoring borders'],
              loc='upper left', ncol=1, frameon=False)
    ax.set_xlim([-100, 1000])
    ax.set_xlabel('Kr$^{78}$ energy (MeV)')
    ax.set_ylabel('Mean gray shades')
    ax.invert_xaxis()  # inverting X axis

    ax_dedx = ax.twinx()
    ax_dedx.plot(kr_dedx['IonEnergy(MeV)'], kr_dedx['dE/dxElec'],
                 linewidth=3, linestyle='--', color='#7570b3')
    ax_dedx.set_ylabel('Electronic dE/dx (keV/$\mu m$)')

    plt.savefig('Fig_10d.eps', bbox_inches='tight')

    return None


def figure_12():
    """
    Figure 12: Tracks separated in Figure 11 using the WUSEM algorithm,
    and then enumerated using the function enumerate_objects(). As in
    Figure 5, the size of the first structuring element is small when
    compared to the objects, and smaller regions where tracks overlap are
    counted as tracks. (a) Considering border tracks. (b) Ignoring border
    tracks. Parameters for WUSEM algorithm: initial_radius = 5,
    delta_radius = 2.
    """

    image = imread('orig_figures/dataset_02/FT-Lab_19.07.390.MAG1.jpg',
                   as_grey=True)

    # Figure 12 (a).
    imgbin_wb = binary_fill_holes(image < threshold_isodata(image))
    imglabel_wb, _, _ = ds.segmentation_wusem(imgbin_wb, initial_radius=5,
                                              delta_radius=2)
    imgnumber_wb = ds.enumerate_objects(image, imglabel_wb, font_size=25)

    plt.figure(figsize=(10, 12))
    plt.imshow(imgnumber_wb, cmap='gray')
    plt.savefig('Fig_12a.eps', bbox_inches='tight')

    # Figure 12 (b).
    imgbin_nb = clear_border(binary_fill_holes(image <
                             threshold_isodata(image)))
    imglabel_nb, _, _ = ds.segmentation_wusem(imgbin_nb, initial_radius=5,
                                              delta_radius=2)
    imgnumber_nb = ds.enumerate_objects(image, imglabel_nb, font_size=25)

    plt.figure(figsize=(10, 12))
    plt.imshow(imgnumber_nb, cmap='gray')
    plt.savefig('Fig_12b.eps', bbox_inches='tight')

    return None


def figure_13():
    """
    Figure 13: Comparison between manual and automatic counting for
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
    plot_colors = {'MAG1': '#472d7b', 'MAG2': '#addc30'}
    color_fit = '#e41a1c'
    auto_best = {}

    man_count = pd.read_excel('manual_count/manual_dataset02.xls')
    auto_count = pd.read_csv('auto_count/auto_dataset02.txt')

    manual, auto, meanman_wb, meanman_nb = [{} for _ in range(4)]
    manual = {'MAG1': man_count.query('image <= 8 or image == 19'),
              'MAG2': man_count.query('image > 8 and image < 18')}

    auto = {'MAG1': auto_count.query('image <= 8 or image == 19'),
            'MAG2': auto_count.query('image > 8 and image < 18')}

    for key, val in auto.items():
        # best candidate for considering and ignoring borders scenarios
        # is the same.
        auto_best[key] = val[(val['initial_radius'] == 10) &
                             (val['delta_radius'] == 8)]

    manvsauto_wb, manvsauto_nb, box_colors = [[] for _ in range(3)]
    pos = list(range(1, 5))

    for _, val in plot_colors.items():
        box_colors.append('0.80')
        box_colors.append(val)

    flier_props = dict(marker='P', markerfacecolor='#386cb0',
                       markeredgecolor='#386cb0', linestyle='none')

    x_ticks = [1.5, 3.5]
    x_labels = ['Magnification 1', 'Magnification 2']

    for key, val in manual.items():
        # data for considering and ignoring borders scenarios.
        manvsauto_wb.append([np.asarray(val.manual_withborder)])
        manvsauto_wb.append([np.asarray(auto_best[key].auto_withborder)])

        manvsauto_nb.append([np.asarray(val.manual_noborder)])
        manvsauto_nb.append([np.asarray(auto_best[key].auto_noborder)])

    # Figure 13 (a).
    fig, ax = plt.subplots(figsize=(12, 12))
    box_plot = ax.boxplot(manvsauto_wb, flierprops=flier_props, positions=pos)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    num_boxes = len(manvsauto_wb)
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
                 [np.average(manvsauto_wb[i])],
                 color='w', marker='P', markeredgecolor='k')

    ax.set_xlabel('Sample number')
    ax.set_ylabel('Tracks counted')

    plt.savefig('Fig_13a.eps', bbox_inches='tight')

    # Figure 13 (b).
    fig, ax = plt.subplots(figsize=(12, 12))

    # preparing fit variables.
    x = np.linspace(0, 100, 1000)
    aux_manual, aux_auto = [[] for _ in range(2)]

    for key, val in manual.items():
        ax.plot(val.manual_withborder,
                auto_best[key].auto_withborder,
                color=plot_colors[key],
                marker='.',
                linestyle='None',
                markersize=18)

        aux_manual.append(val.manual_withborder.tolist())
        aux_auto.append(auto_best[key].auto_withborder.tolist())

    aux_manual = list(chain.from_iterable(aux_manual))
    aux_auto = list(chain.from_iterable(aux_auto))

    # fitting a line in the data.
    fit = np.polyfit(aux_manual, aux_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(aux_manual, fit_fn(aux_manual), linewidth=3, color=color_fit)
    ax.plot(x, x, '--', color='k')

    # setting axes and labels.
    ax.axis([0, 100, 0, 100])
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')

    # preparing legend.
    sample = []

    for key, var in plot_colors.items():
        sample.append(mpatches.Patch(color=var,
                                     label='Magnification ' + str(key)[-1]))

    ax.legend(handles=[sample[0], sample[1]],
              loc='lower right', ncol=1, frameon=False)

    plt.savefig('Fig_13b.eps', bbox_inches='tight')

    # Figure 13 (c).
    fig, ax = plt.subplots(figsize=(12, 12))
    box_plot = ax.boxplot(manvsauto_nb, flierprops=flier_props, positions=pos)

    ax.set_xticks(x_ticks)
    ax.set_xticklabels(x_labels)

    num_boxes = len(manvsauto_nb)
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
                 [np.average(manvsauto_nb[i])],
                 color='w', marker='P', markeredgecolor='k')

    ax.set_xlabel('Sample number')
    ax.set_ylabel('Tracks counted')

    plt.savefig('Fig_13c.eps', bbox_inches='tight')

    # Figure 13 (d).
    fig, ax = plt.subplots(figsize=(12, 12))

    # preparing fit variables.
    x = np.linspace(0, 100, 1000)
    aux_manual, aux_auto = [[] for _ in range(2)]

    for key, val in manual.items():
        ax.plot(val.manual_noborder,
                auto_best[key].auto_noborder,
                color=plot_colors[key],
                marker='.',
                linestyle='None',
                markersize=18)

        aux_manual.append(val.manual_noborder.tolist())
        aux_auto.append(auto_best[key].auto_noborder.tolist())

    aux_manual = list(chain.from_iterable(aux_manual))
    aux_auto = list(chain.from_iterable(aux_auto))

    # fitting a line in the data.
    fit = np.polyfit(aux_manual, aux_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(aux_manual, fit_fn(aux_manual), linewidth=3, color=color_fit)
    ax.plot(x, x, '--', color='k')

    # setting axes and labels.
    ax.axis([0, 100, 0, 100])
    ax.set_xlabel('Manual counting')
    ax.set_ylabel('Automatic counting')

    # preparing legend.
    sample = []

    for key, var in plot_colors.items():
        sample.append(mpatches.Patch(color=var,
                                     label='Magnification ' + str(key)[-1]))

    ax.legend(handles=[sample[0], sample[1]],
              loc='lower right',
              ncol=1,
              frameon=False)

    plt.savefig('Fig_13d.eps', bbox_inches='tight')

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

    auto = pd.read_csv('auto_count/autoincid_Kr-78_4,5min.txt')
    datak90_image1 = auto[(auto['folder'] == 'K90_incid') &
                          (auto['image'] == 1)]

    # initial_radius starts in 5, ends in 40 and has delta_radius 5.
    # delta_radius starts in 2, ends in 20 and has delta_radius 2.
    # let's create matrices to accomodate the countings.
    XX, YY = np.mgrid[5:41:5, 2:21:2]
    ZZk90_wb, ZZk90_nb = np.zeros(XX.shape), np.zeros(XX.shape)

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        aux = int(datak90_image1.auto_withborder[(auto.initial_radius == i) &
                                                 (auto.delta_radius == j)])
        ZZk90_wb[(XX == i) & (YY == j)] = aux

        aux = int(datak90_image1.auto_noborder[(auto.initial_radius == i) &
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
                   'K90_incid4,5min_1_track_11.eps',
                   'K90_incid4,5min_1_track_14.eps',
                   'K90_incid4,5min_1_track_15.eps',
                   'K90_incid4,5min_1_track_21.eps',
                   'K90_incid4,5min_1_track_22.eps',
                   'K90_incid4,5min_1_track_23.eps',
                   'K90_incid4,5min_1_track_24.eps']

    # Supplementary Figure 2, from (a) to (j).
    output_files = ['Fig_sup2a.eps', 'Fig_sup2b.eps', 'Fig_sup2c.eps',
                    'Fig_sup2d.eps', 'Fig_sup2e.eps', 'Fig_sup2f.eps',
                    'Fig_sup2g.eps', 'Fig_sup2h.eps', 'Fig_sup2i.eps',
                    'Fig_sup2j.eps']

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
    plt.figure(figsize=(10, 12))
    plt.imshow(imgbin_wb, cmap='gray')
    plt.savefig('Fig_sup3a.eps', bbox_inches='tight')

    # Supplementary  Figure 3 (b).
    plt.figure(figsize=(10, 12))
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
        aux = int(image.auto_withborder[(auto_set2.initial_radius == i) &
                                        (auto_set2.delta_radius == j)])
        ZZ_wb[(XX == i) & (YY == j)] = aux

        aux = int(image.auto_noborder[(auto_set2.initial_radius == i) &
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
        meanman_wb[key] = val.manual_withborder.mean()
        # without border.
        meanman_nb[key] = val.manual_noborder.mean()

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
            aux_wb = val.auto_withborder[(val.initial_radius == i) &
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
            aux_nb = val.auto_noborder[(val.initial_radius == i) &
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
