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

from itertools import product
from matplotlib.ticker import MultipleLocator
from scipy.ndimage.morphology import binary_fill_holes
from skimage.color import rgb2gray
from skimage.filters import threshold_isodata
from skimage.io import ImageCollection, imread, imread_collection
from skimage.segmentation import clear_border

import desiqueira2017 as ds  # functions presented in this study
import matplotlib.pyplot as plt
import numpy as np
import os
import pandas as pd
import statistics as stats


def generating_results_dataset1(base_path='.', save_images=False):
    """Process every image on dataset 1 using WUSEM, initial_radius will
    vary from 5 to 40, step 5, and delta_radius from 2 to 20, step 2.
    Please keep in mind that this command will take **several hours** to
    finish.

    Parameters
    ----------
    base_path : string
        The path where the folders orig_figures and res_figures are.
    save_images : bool, optional (default : False)
        If True, the resulting images will be written to the disk. Else,
        they will only be presented.

    Returns
    -------
    None

    Examples
    --------
    >>> generating_results_dataset1(base_path='.', save_images=True)
    """

    # establishing the variables to be used.
    frad_num = range(5, 41, 5)
    delrad_num = range(2, 21, 2)
    path_num = [0, 20, 30, 40, 50, 60, 70, 80, 90]
    minutes = ['4,5min', '8,5min']

    for m in minutes:
        orig_path = 'orig_figures/dataset_01/Kr-78_' + m + '/'
        save_path = 'res_figures/dataset_01/Kr-78_' + m + '/'

        # preparing the file to receive data.
        file = open(base_path + 'autoincid_Kr-78_' + m + '_incid.txt',
                    'w')
        file.write(('folder,image,initial_radius,delta_radius,'
                    'auto_withborder,auto_noborder\n'))

        os.chdir(base_path)

        for num in path_num:
            folder = 'K' + str(num) + '_incid'
            os.chdir(base_path + orig_path + folder)
            image_set = imread_collection(load_pattern='*.bmp')

            for idx, image in enumerate(image_set):
                for fr, dr in product(frad_num, delrad_num):
                    num_aux = []
                    # counting with border: True, without border: False.
                    for count in [True, False]:
                        num_aux.append(wusem_results(image,
                                                     initial_radius=fr,
                                                     delta_radius=dr,
                                                     count_border=count,
                                                     save_images=save_images))
                        if save_images:
                            img_filename = base_path + save_path + \
                                           folder + '/' + folder + \
                                           '_' + str(idx+1) + '_fr' + \
                                           str(fr) + '_dr' + str(dr) + \
                                           '_' + str(count)[0] + '.png'
                            os.rename(src='resulting_image.png',
                                      dst=img_filename)

                    # saving results in a file.
                    line_file = folder + ',' + str(idx+1) + ',' + \
                        str(fr) + ',' + str(dr) + ',' + \
                        str(num_aux[0]) + ',' + \
                        str(num_aux[1]) + '\n'
                    file.write(line_file)

        file.close()

    return None


def generating_results_dataset2(base_path='.', save_images=False):
    """Process every image on dataset 2 using WUSEM, initial_radius will
    vary from 5 to 40, step 5, and delta_radius from 2 to 20, step 2.
    Please keep in mind that this command will take a while to finish.

    Parameters
    ----------
    base_path : string
        The path where the folders orig_figures and res_figures are.
    save_images : bool, optional (default : False)
        If True, the resulting images will be written to the disk. Else,
        they will only be presented.

    Returns
    -------
    None

    Examples
    --------
    >>> generating_results_dataset2(base_path='.', save_images=True)
    """

    # establishing the variables to be used.
    fst_num = range(5, 41, 5)
    delta_radius_num = range(2, 21, 2)

    orig_path = 'orig_figures/dataset_02/'
    save_path = 'res_figures/dataset_02/'

    # preparing the file to receive data.
    file = open(base_path + 'automatic_DAP.txt', 'w')
    file.write(('image,initial_radius,delta_radius,auto_withborder,'
                'auto_noborder\n'))

    os.chdir(base_path + orig_path)
    image_set = ImageCollection(load_pattern='*.jpg',
                                load_func=imread_convert)
    for image in image_set:
        image = rgb2gray(image)

    for idx, image in enumerate(image_set):
        for fs, st in product(fst_num, delta_radius_num):
            num_aux = []
            for count in [True, False]:
                # counting with border: True, without border: False.
                num_aux.append(show_schaller_results(image,
                                                     initial_radius=fs,
                                                     delta_radius=st,
                                                     count_border=count,
                                                     save_images=save_images))
                if save_images:
                    img_filename = base_path + save_path + 'img' + \
                                   str(idx+1) + '_fs' + str(fs) + \
                                   '_st' + str(st) + '_' + \
                                   str(count)[0] + '.png'
                    os.rename(src='resulting_image.png',
                              dst=img_filename)

                # saving results in a file.
            line_file = str(idx+1) + ',' + str(fs) + ',' + str(st) + \
                ',' + str(num_aux[0]) + ',' + str(num_aux[1]) + \
                '\n'
            file.write(line_file)

    file.close()

    return None


def imread_convert(image):
    """Support function for ImageCollection. Converts the input image to
    gray.
    """

    return imread(image, as_grey=True)


def mean_efficiency():
    """
    """

    manual_45 = pd.read_excel('manual_count/manual_Kr-78_4,5min.xls')
    manual_85 = pd.read_excel('manual_count/manual_Kr-78_8,5min.xls')
    manual_dat2 = pd.read_excel('manual_count/manual_dataset02.xls')

    auto_45 = pd.read_csv('auto_count/autoincid_Kr-78_4,5min.txt')
    auto_85 = pd.read_csv('auto_count/autoincid_Kr-78_8,5min.txt')
    auto_dat2 = pd.read_csv('auto_count/auto_dataset02.txt')

    auto = np.asarray(list(auto_45[(auto_45['initial_radius'] == 5) &
                                   (auto_45['delta_radius'] == 4)].auto_withborder) +
                      list(auto_45[(auto_45['initial_radius'] == 25) &
                                   (auto_45['delta_radius'] == 2)].auto_noborder) +
                      list(auto_85[(auto_85['initial_radius'] == 5) &
                                   (auto_85['delta_radius'] == 4)].auto_withborder) +
                      list(auto_85[(auto_85['initial_radius'] == 25) &
                                   (auto_85['delta_radius'] == 2)].auto_noborder) +
                      list(auto_dat2[(auto_dat2['initial_radius'] == 10) &
                                     (auto_dat2['delta_radius'] == 8)].auto_withborder) +
                      list(auto_dat2[(auto_dat2['initial_radius'] == 10) &
                                     (auto_dat2['delta_radius'] == 8)].auto_noborder))

    manual = np.asarray(list(manual_45.manual_withborder) + list(manual_45.manual_noborder) +
                    list(manual_85.manual_withborder) + list(manual_85.manual_noborder) +
                    list(manual_dat2.manual_withborder) + list(manual_dat2.manual_noborder))

    smp_mean = np.mean(auto / manual)
    smp_stdmean = np.std(auto / manual) / np.sqrt(len(auto))

    print('mu = ', smp_mean, ', sigma = ', smp_stdmean)

    return None


def plot_auxiliar_dataset1(var_manual, var_auto, auto_color='b',
                           ax=None):
    """Auxiliar bar plots for comparing manual and automatic counting
    (dataset 1). Also presents statistics.
    """

    if ax is None:
        ax = plt.gca()

    width = 0.5

    rows = len(var_manual)
    x_trk = np.arange(0, rows)
    # manual counting.
    ax.bar(left=x_trk-width, height=var_manual, width=width,
           color='0.60')
    # automatic counting.
    ax.bar(left=x_trk, height=var_auto, width=width, color=auto_color)

    ax.axis([-1, rows, 0, 45])
    ax.set_xticks([0, 4, 9, 14, 19])
    ax.set_xticklabels(('1', '5', '10', '15', '20'))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    x_label = ('Image number\n\nManual:\n* Sample st dev: ' +
               str(np.round(stats.stdev(var_manual), decimals=2)) +
               '\n* Pop st dev: ' +
               str(np.round(np.std(var_manual), decimals=2)) +
               '\n* Poisson variance: ' +
               str(np.round(np.sqrt(np.mean(var_manual)),
                            decimals=2)) +
               '\n* SSD / PV: ' +
               str(np.round(stats.stdev(var_manual) /
                            np.sqrt(np.mean(var_manual)),
                            decimals=2)) +
               '\n\nAutomatic:\n* Sample st dev: ' +
               str(np.round(stats.stdev(var_auto), decimals=2)) +
               '\n* Pop st dev: ' +
               str(np.round(np.std(var_auto), decimals=2)) +
               '\n* Poisson variance: ' +
               str(np.round(np.sqrt(np.mean(var_auto)),
                            decimals=2)) +
               '\n* SSD / PV: ' +
               str(np.round(stats.stdev(var_auto) /
                            np.sqrt(np.mean(var_auto)),
                            decimals=2)))

    ax.set_xlabel(x_label)
    ax.set_ylabel('Tracks counted')

    return ax


def plot_auxiliar_dataset2(var_manual, var_auto, auto_color='b',
                           ax=None):
    """Auxiliar bar plots for comparing manual and automatic counting
    (dataset 2). Also presents statistics.
    """

    if ax is None:
        ax = plt.gca()

    width = 0.4

    rows = len(var_manual)
    x_trk = np.arange(0, rows)
    # manual counting.
    ax.bar(left=x_trk-width, height=var_manual, width=width,
           color='0.60')
    # automatic counting.
    ax.bar(left=x_trk, height=var_auto, width=width, color=auto_color)

    ax.set_xticks(x_trk - width / 2)
    ax.axis([-1, rows, 0, 100])
    ax.set_xticklabels(('1', '2', '3', '4', '5', '6', '7', '8', '9'))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    x_label = ('Image number\n\nManual:\n* Sample st dev: ' +
               str(np.round(stats.stdev(var_manual), decimals=2)) +
               '\n* Pop st dev: ' +
               str(np.round(np.std(var_manual), decimals=2)) +
               '\n* Poisson variance: ' +
               str(np.round(np.sqrt(np.mean(var_manual)),
                            decimals=2)) +
               '\n* SSD / PV: ' +
               str(np.round(stats.stdev(var_manual) /
                            np.sqrt(np.mean(var_manual)),
                            decimals=2)) +
               '\n\nAutomatic:\n* Sample st dev: ' +
               str(np.round(stats.stdev(var_auto), decimals=2)) +
               '\n* Pop st dev: ' +
               str(np.round(np.std(var_auto), decimals=2)) +
               '\n* Poisson variance: ' +
               str(np.round(np.sqrt(np.mean(var_auto)),
                            decimals=2)) +
               '\n* SSD / PV: ' +
               str(np.round(stats.stdev(var_auto) /
                            np.sqrt(np.mean(var_auto)),
                            decimals=2)))

    ax.set_xlabel(x_label)
    ax.set_ylabel('Tracks counted')

    return ax


def plot_linreg_dataset1(var_manual, var_auto, color='red', ax=None):
    """Auxiliar linear regression for comparing manual and automatic
    counting (dataset 1). Also presents statistics.
    """

    if ax is None:
        ax = plt.gca()

    # identity function.
    x = np.linspace(0, 45, 1000)
    ax.plot(x, x, 'k--')

    # linear regression.
    fit = np.polyfit(var_manual, var_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(var_manual, var_auto, color=color, marker='.',
            linestyle='None', markersize=10)
    ax.plot(x, fit_fn(x), color=color)

    # statistics.
    corr_coef = np.corrcoef(var_manual, var_auto)[0, 1]
    covariance = np.cov(var_manual, var_auto)[0, 1]

    x_label = ('Manual counting\n\nLine parameters:\n* Slope: ' +
               str(np.round(fit[0], decimals=2)) +
               '\n* Y-Intercept:\n\n' +
               str(np.round(fit[1], decimals=2)) +
               '\n\nStats:\n* Correlation: ' +
               str(np.round(corr_coef, decimals=2)) +
               '\n* Covariance: ' +
               str(np.round(covariance, decimals=2)) +
               '\n* Manual / Auto ratio: ' +
               str(np.round(np.mean(var_manual)/np.mean(var_auto),
                            decimals=2)))

    ax.set_xlabel(x_label)
    ax.set_ylabel('Automatic counting')

    ax.xaxis.set_major_locator(MultipleLocator(10))
    ax.yaxis.set_major_locator(MultipleLocator(10))

    return ax


def plot_linreg_dataset2(var_manual, var_auto, color='red', ax=None):
    """Auxiliar linear regression for comparing manual and automatic
    counting (dataset 2). Also presents statistics.
    """

    if ax is None:
        ax = plt.gca()

    # identity function.
    x = np.linspace(0, 100, 1000)
    ax.plot(x, x, 'k--')

    # linear regression.
    fit = np.polyfit(var_manual, var_auto, deg=1)
    fit_fn = np.poly1d(fit)
    ax.plot(var_manual, var_auto,
            color=color, marker='.',
            linestyle='None',
            markersize=10)
    ax.plot(x, fit_fn(x), color=color)

    # statistics.
    corr_coef = np.corrcoef(var_manual, var_auto)[0, 1]
    covariance = np.cov(var_manual, var_auto)[0, 1]

    x_label = ('Manual counting\n\nLine parameters:\n* Slope: ' +
               str(np.round(fit[0], decimals=2)) +
               '\n* Y-Intercept:\n\n' +
               str(np.round(fit[1], decimals=2)) +
               '\n\nStats:\n* Correlation: ' +
               str(np.round(corr_coef, decimals=2)) +
               '\n* Covariance: ' +
               str(np.round(covariance, decimals=2)) +
               '\n* Manual / Auto ratio: ' +
               str(np.round(np.mean(var_manual)/np.mean(var_auto),
                            decimals=2)))

    ax.set_xlabel(x_label)
    ax.set_ylabel('Automatic counting')

    return ax


def wusem_results(image, initial_radius=10, delta_radius=5,
                  count_border=True, save_images=False):
    """Support function. Process image according to initial_radius and
    delta_radius, and presents the results of ISODATA threshold, WUSEM
    algorithm, and enumerate_objects. Returns the nu

    Parameters
    ----------
    image : string
        The path where the folders orig_figures and res_figures are.
    initial_radius : float, optional (default : 10)
        Radius of the first structuring element, in pixels.
    delta_radius : float, optional (default : 5)
        Size of the radius to be used on each iteration, in pixels.
    count_border : bool, optional(default : True)
        Chooses whether to use the scenario 'considering track borders'
        (True) or 'ignoring track borders' (False).
    save_images : bool, optional (default : False)
        If True, the resulting images will be written to the disk. Else,
        they will only be presented.

    Returns
    -------
    num_objects : int
        Number of objects in the input image.
    """

    thresh = threshold_isodata(image)
    img_bin = binary_fill_holes(image < thresh)

    if not count_border:
        img_bin = clear_border(img_bin)

    img_labels, num_objects, _ = ds.segmentation_wusem(img_bin,
                                                       initial_radius=initial_radius,
                                                       delta_radius=delta_radius)
    img_number = ds.enumerate_objects(image, img_labels, font_size=25)

    fig, ax = plt.subplots(nrows=2, ncols=2, figsize=(16, 12))

    ax[0, 0].imshow(image, cmap='gray')
    ax[0, 1].imshow(img_bin, cmap='gray')
    ax[1, 0].imshow(img_labels, cmap='nipy_spectral')
    ax[1, 1].imshow(img_number)

    if save_images:
        plt.savefig(filename='resulting_image.png', bbox_inches='tight')
        plt.close('all')
    else:
        plt.show()

    return num_objects
