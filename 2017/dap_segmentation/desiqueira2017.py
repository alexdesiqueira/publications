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
Supplementary Material'.  If not, see <http://www.gnu.org/licenses/>.
"""

from itertools import product
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage.morphology import (binary_fill_holes,
                                      distance_transform_edt)
from scipy.stats import norm
from skimage import morphology
from skimage.color import gray2rgb
from skimage.io import imread, imread_collection
from skimage.filters import threshold_isodata
from skimage.measure import regionprops, label
from skimage.morphology import remove_small_objects
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte
from sys import platform

import matplotlib.mlab as mlab
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

# Defining the file extension to save all generated images.
FILE_EXT = '.jpg'


def all_round_regions(sample_set, initial_radius=25, delta_radius=2,
                      eccentricity=0.3, count_border=True):
    """Returns and saves data from round objects within each image of
    dataset 1.

    Parameters
    ----------
    sample_set : string
        Sample set to analyze. Accepts the values 'Kr-78_4,5min' and
        'Kr-78_8,5min'.
    initial_radius : float, optional (default : 25)
        Radius of the first structuring element, in pixels.
    delta_radius : float, optional (default : 2)
        Size of the radius to be used on each iteration, in pixels.
    eccentricity : float, optional (default : 0.3)
        Defines the eccentricity tolerance. The region has to be an
        eccentricity smaller than this value to be considered a track.
    count_border : bool, optional(default : True)
        Chooses whether to use the scenario 'considering track borders'
        (True) or 'ignoring track borders' (False).

    Returns
    -------
    None

    Examples
    --------
    >>> all_round_regions('Kr-78_4,5min', first_step=5, step=4,
                          eccentricity=0.3, count_border=True)
    >>> all_round_regions('Kr-78_8,5min', first_step=20, step=5,
                          eccentricity=0.45, count_border=True)
    """

    folders = ['K0_incid', 'K20_incid', 'K30_incid', 'K40_incid',
               'K50_incid', 'K60_incid', 'K70_incid', 'K80_incid',
               'K90_incid']

    for sample in sample_set:
        file = open('roundinfo_' + sample + '_' + str(count_border)[0] +
                    '.txt', 'w')
        header = 'folder,image,object,minor_axis,major_axis,mean_gray\n'
        file.write(header)

        for folder in folders:
            pattern = 'orig_figures/dataset_01/' + sample + '/' + \
                      str(folder) + '/*.bmp'
            image_set = imread_collection(load_pattern=pattern)

            for idx, image in enumerate(image_set):
                _, _, info_reg = round_regions(image,
                                               initial_radius=initial_radius,
                                               delta_radius=delta_radius,
                                               toler_ecc=eccentricity,
                                               count_border=count_border)

                for info in info_reg:
                    line_file = (folder + ',' + str(idx+1) + ',' +
                                 str(info[0]) + ',' + str(info[1]) +
                                 ',' + str(info[2]) + ',' +
                                 str(info[3]) + '\n')
                    file.write(line_file)

        file.close()

    return None


def clean_track_info(data, sigma_number=2, save_file=False):
    """Returns and saves data from round objects within each image of
    dataset 1.

    Parameters
    ----------
    data : DataFrame
        Data to limit.
    sigma_number : float, optional (default : 2)
        The standard deviation used to limit the data.
    save_file : bool, optional (default : False)
        If True, saves the results in the file 'roundclean.txt'.

    Returns
    -------
    data_clean : DataFrame
        DataFrame containing only the data within the interval
        [mean - std, mean + std].

    Examples
    --------
    >>> import pandas as pd
    >>> info_tracks = pd.read_csv('auto_count/roundinfo_Kr-78_4,5min_T.txt')
    >>> info_clean = clean_track_info(info_tracks)

    """

    inf_lim = data['minor_axis'].mean() - sigma_number*data['minor_axis'].std()
    sup_lim = data['minor_axis'].mean() + sigma_number*data['minor_axis'].std()

    data_clean = data[(data['minor_axis'] >= inf_lim) &
                      (data['minor_axis'] <= sup_lim)]

    if save_file:
        data_clean.to_csv('roundclean.txt')

    return data_clean


def comparison_counting(manual, auto, with_border=False, tolerance=2.5):
    """
    """

    step_cand = []

    for i, j in product(range(5, 41, 5), range(2, 21, 2)):
        if with_border:
            aux_auto = auto.auto_withborder[(auto.initial_radius == i) &
                                            (auto.delta_radius == j)].mean()
        else:
            aux_auto = auto.auto_noborder[(auto.initial_radius == i) &
                                          (auto.delta_radius == j)].mean()
        if 0 < (manual - aux_auto) < tolerance:
            step_cand.append([i, j])
    return step_cand


def enumerate_objects(image, labels, font_size=30):
    """Generate an image with each labeled region numbered.

    Parameters
    ----------
    labels : (rows, cols) ndarray
        Labeled image.
    equal_intensity : bool, optional (default : True)
        If True, each region on the output image will have the same
        intensity.
    font_size : int, optional (default : 30)
        Font size to be used when numbering.

    Returns
    -------
    img_numbered : (rows, cols) ndarray
        Labeled image with each region numbered.

    Examples
    --------
    >>> from skcv.draw import draw_synthetic_circles
    >>> from skimage.measure import label
    >>> image = draw_synthetic_circles((512, 512), quant=25, seed=0)
    >>> img_label = label(image)
    >>> img_numbered = enumerate_objects(image,
                                         img_label,
                                         font_size=18)
    """

    # avoiding original image to be modified.
    aux_image = np.copy(gray2rgb(image))

    # obtaining info from the objects.
    obj_info = []
    for idx in regionprops(labels):
        obj_info.append([idx.centroid[0],
                         idx.centroid[1],
                         str(idx.label)])

    # default fonts to be used on each system.
    if platform.startswith('linux'):
        font_name = '/usr/share/fonts/truetype\
                     /liberation/LiberationSans-Bold.ttf'
    elif platform.startswith('win'):
        font_name = 'c:/windows/fonts/arialbd.ttf'
    '''
    # Please MacOS user, uncomment these lines and fill a font path.
    elif platform.startswith('darwin'):
        font_name = ''
    '''
    font = ImageFont.truetype(font_name, font_size)

    img_numbered = Image.fromarray(img_as_ubyte(aux_image))
    draw = ImageDraw.Draw(img_numbered)

    # drawing numbers on each region.
    for obj in obj_info:
        draw.text((obj[1], obj[0]), obj[2], fill=(255, 255, 0),
                  font=font)

    return img_numbered


def joining_candidates(dict_cand):
    """Extracts candidates in a dict, returning a list.

    Parameters
    ----------
    dict_cand : dict
        Candidates in a dictionary.

    Returns
    -------
    list_cand : list
        Candidates in a list.
    """

    list_cand = []

    for _, value in dict_cand.items():
        list_cand += value

    return list_cand


def parameters_samples(var):
    """Helping function. Returns the statistics of a variable in a tuple."""

    var_param = [np.mean(var), np.std(var)]

    return var_param


def px_to_um(px=25):
    """Helping function. Converts pixels to um according to the microscopy
    magnification."""

    ang_coef = 0.41402227
    lin_coef = 0.05954835

    um = px*ang_coef + lin_coef

    return um


def px_to_um2(diam_a=20, diam_b=25):
    """Helping function. Converts pixels to um^2 according to the microscopy
    magnification."""


    ang_coef = 0.41402227
    lin_coef = 0.05954835

    um2 = (diam_a*ang_coef + lin_coef) * (diam_b*ang_coef + lin_coef)

    return um2


def ratio_manauto(manual, auto, sample='kr', count_border=True,
                  save_figure=False):

    all_ratio = []

    if sample is 'dap':
        rows, cols = 1, 2
        plot_which = {'[0]': 'MAG1', '[1]': 'MAG2'}

        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 8))

        for j in range(cols):
            if count_border:
                ratio = np.asarray(auto[plot_which[str([j])]].auto_withborder) / \
                        np.asarray(manual[plot_which[str([j])]].manual_withborder)
            else:
                ratio = np.asarray(auto[plot_which[str([j])]].auto_noborder) / \
                        np.asarray(manual[plot_which[str([j])]].manual_noborder)

            (mu, sigma) = norm.fit(ratio)
            n, bins, patches = ax[j].hist(ratio, bins=6, normed=True,
                                          edgecolor='k')
            fit = mlab.normpdf(bins, mu, sigma)
            ax[j].plot(bins, fit, 'k--', linewidth=2)
            ax[j].set_xlabel('$\mu$: ' + str(np.round(ratio.mean(),
                                                      decimals=4)) +
                             ', $\sigma$: ' + str(np.round(ratio.std(),
                                                           decimals=4)))

            all_ratio.append([plot_which[str([j])],
                              ratio.mean(),
                              ratio.std()])

    else:
        rows, cols = 3, 3
        plot_which = {'[0, 0]': '0', '[0, 1]': '20', '[0, 2]': '30',
                      '[1, 0]': '40', '[1, 1]': '50', '[1, 2]': '60',
                      '[2, 0]': '70', '[2, 1]': '80', '[2, 2]': '90'}

        fig, ax = plt.subplots(nrows=rows, ncols=cols, figsize=(15, 20))

        for i, j in product(range(rows), range(cols)):
            if count_border:
                ratio = np.asarray(auto[plot_which[str([i, j])]].auto_withborder) / \
                        np.asarray(manual[plot_which[str([i, j])]].manual_withborder)
            else:
                ratio = np.asarray(auto[plot_which[str([i, j])]].auto_noborder) / \
                        np.asarray(manual[plot_which[str([i, j])]].manual_noborder)

            (mu, sigma) = norm.fit(ratio)
            n, bins, patches = ax[i, j].hist(ratio, normed=True, edgecolor='k')
            fit = mlab.normpdf(bins, mu, sigma)
            ax[i, j].plot(bins, fit, 'k--', linewidth=2)
            ax[i, j].set_xlabel('$\mu$: ' + str(np.round(ratio.mean(),
                                                         decimals=4)) +
                                ', $\sigma$: ' + str(np.round(ratio.std(),
                                                              decimals=4)))

            all_ratio.append([plot_which[str([i, j])],
                              ratio.mean(),
                              ratio.std()])

    if save_figure:
        plt.savefig('ratio_manauto' + FILE_EXT)
    else:
        plt.show()

    return all_ratio


def round_regions(image, initial_radius=25, delta_radius=2, toler_ecc=0.5,
                  count_border=True):
    '''
    '''

    info_regions = []
    img_objects = np.zeros(image.shape)

    thresh = threshold_isodata(image)
    img_bin = binary_fill_holes(image < thresh)
    img_bin = morphology.remove_small_objects(img_bin)

    if not count_border:
        img_bin = clear_border(img_bin)

    img_labels, _, _ = segmentation_wusem(img_bin,
                                          initial_radius=initial_radius,
                                          delta_radius=delta_radius)
    properties = regionprops(img_labels, intensity_image=image)

    for prop in properties:
        if prop.eccentricity > toler_ecc:
            img_labels[img_labels == prop.label] = 0
        else:
            info_regions.append([prop.label,
                                 prop.minor_axis_length,
                                 prop.major_axis_length,
                                 prop.mean_intensity])

    rows, cols = np.where(img_labels != 0)
    img_objects[rows, cols] = image[rows, cols]

    return img_labels, img_objects, info_regions


def segmentation_wusem(image, str_el='disk', initial_radius=10,
                       delta_radius=5):
    """Separates regions on a binary input image using successive
    erosions as markers for the watershed algorithm. The algorithm stops
    when the erosion image does not have objects anymore.

    Parameters
    ----------
    image : (N, M) ndarray
        Binary input image.
    str_el : string, optional
        Structuring element used to erode the input image. Accepts the
        strings 'diamond', 'disk' and 'square'. Default is 'disk'.
    initial_radius : int, optional
        Initial radius of the structuring element to be used in the
        erosion. Default is 10.
    delta_radius : int, optional
        Delta radius used in the iterations:
         * Iteration #1: radius = initial_radius + delta_radius
         * Iteration #2: radius = initial_radius + 2 * delta_radius,
        and so on. Default is 5.

    Returns
    -------
    img_labels : (N, M) ndarray
        Labeled image presenting the regions segmented from the input
        image.
    num_objects : int
        Number of objects in the input image.
    last_radius : int
        Radius size of the last structuring element used on the erosion.

    References
    ----------
    .. [1] F.M. Schaller et al. "Tomographic analysis of jammed ellipsoid
    packings", in: AIP Conference Proceedings, 2013, 1542: 377-380. DOI:
    10.1063/1.4811946.

    Examples
    --------
    >>> from skimage.data import binary_blobs
    >>> image = binary_blobs(length=512, seed=0)
    >>> img_labels, num_objects, _ = segmentation_wusem(image,
                                                        str_el='disk',
                                                        initial_radius=10,
                                                        delta_radius=3)
    """

    rows, cols = image.shape
    img_labels = np.zeros((rows, cols))
    curr_radius = initial_radius
    distance = distance_transform_edt(image)

    while True:
        aux_se = {
            'diamond': morphology.diamond(curr_radius),
            'disk': morphology.disk(curr_radius),
            'square': morphology.square(curr_radius)
        }
        str_el = aux_se.get('disk', morphology.disk(curr_radius))

        erod_aux = morphology.binary_erosion(image, selem=str_el)
        if erod_aux.min() == erod_aux.max():
            last_step = curr_radius
            break

        markers = label(erod_aux)
        curr_labels = morphology.watershed(-distance,
                                           markers,
                                           mask=image)
        img_labels += curr_labels

        # preparing for another loop.
        curr_radius += delta_radius

    # reordering labels.
    img_labels = label(img_labels)

    # removing small labels.
    img_labels, num_objects = label(remove_small_objects(img_labels),
                                    return_num=True)

    return img_labels, num_objects, last_step


def separate_tracks_set1(minutes='4,5min', folder='K90_incid', img_number=1,
                         best_args=(5, 4), count_border=True, save_tracks=False):
    """Uses WUSEM to separate tracks in a folder from the first set.

    Parameters
    ----------
    minutes : string, optional
        Amount of minutes the image was etched. Possible values are '4,5min' and
        '8,5min'. Default is '4,5min'.
    folder : string, optional
        Folder where the image is. Possible values are 'K0_incid', 'K20_incid',
        ..., up to 'K90_incid'. Default is 'K90_incid'.
    img_number : int, optional
        Number of the image in the folder. Default is 1.
    best_args : tuple, optional
        Initial and delta radius to process this image. Default is (5, 4).
    count_border : bool, optional
        Whether the algorithm will consider counting the border or not. Default
        is True.
    save_tracks : bool, optional
        Whether the algorithm will save the processed tracks into the disk.
        Default is False.

    Returns
    -------
    None

    References
    ----------
    .. [1] F.M. Schaller et al. "Tomographic analysis of jammed ellipsoid
    packings", in: AIP Conference Proceedings, 2013, 1542: 377-380. DOI:
    10.1063/1.4811946.

    Examples
    --------
    >>> from skimage.data import binary_blobs
    >>> image = binary_blobs(length=512, seed=0)
    >>> img_labels, num_objects, _ = segmentation_wusem(image,
                                                        str_el='disk',
                                                        initial_radius=10,
                                                        delta_radius=3)
    """

    img_name = 'orig_figures/dataset_01/Kr-78_' + minutes + '/' + folder + '/' + \
               folder + minutes + '_' + str(img_number) + '.bmp'

    image = imread(img_name, as_grey=True)
    labels, objects, _ = round_regions(image, initial_radius=best_args[0],
                                       delta_radius=best_args[1],
                                       toler_ecc=0.3, count_border=True)

    data_name = 'auto_count/roundclean_Kr-78_' + minutes + '_' + \
                str(count_border)[0] + '.txt'
    info_clean = pd.read_csv(data_name, index_col=0)

    track_info = info_clean[(info_clean['folder'] == folder) &
                            (info_clean['image'] == img_number)]
    props = regionprops(labels, intensity_image=image)

    for prop in props:
        if prop.label in np.asarray(track_info['object']):
            fig, ax = plt.subplots()
            img_cont = ax.contour(prop.intensity_image, colors='w')
            ax.clabel(img_cont, fmt='%i', fontsize=12, colors='w')
            ax.contourf(prop.intensity_image, cmap='magma')

            if save_tracks:
                track_name = folder + minutes + '_' + str(img_number) + \
                             '_track_' + str(prop.label) + FILE_EXT
                plt.savefig(filename=track_name, bbox_inches='tight')

    return None


def separate_tracks_set2(img_number=1, best_args=(10, 8),
                         count_border=True, save_tracks=False):

    info_name = pd.read_csv('orig_figures/dataset_02/software_numbering.txt')
    img_name = info_name.loc[info_name['image_number'] == img_number,
                             'corresp_image'].iloc[0]

    image = imread('orig_figures/dataset_02/' + img_name + FILE_EXT,
                   as_grey=True)
    labels, objects, _ = round_regions(image, initial_radius=best_args[0],
                                       delta_radius=best_args[1],
                                       toler_ecc=0.3, count_border=True)

    data_name = 'auto_count/roundclean_dataset02_' + str(count_border)[0] + \
                '.txt'

    info_clean = pd.read_csv(data_name, index_col=0)
    track_info = info_clean[info_clean['image'] == img_number]
    props = regionprops(labels, intensity_image=image)

    for prop in props:
        if prop.label in np.asarray(track_info['object']):
            fig, ax = plt.subplots()
            img_cont = ax.contour(prop.intensity_image, colors='w')
            ax.clabel(img_cont, fmt='%.2f', fontsize=12, colors='w')
            ax.contourf(prop.intensity_image, cmap='magma')

            if save_tracks:
                track_name = img_name + '_track_' + str(prop.label) + FILE_EXT
                plt.savefig(filename=track_name, bbox_inches='tight')

    return None


def _sorting_candidates(list_candidates):
    """Helping function. Sorts candidate numbers within a list."""

    aux, counted_list = [], []

    for element in list_candidates:
        if element not in aux:
            aux.append(element)
            counted_list.append([element,
                                 list_candidates.count(element)])

    return sorted(counted_list, reverse=True, key=lambda count: count[1])


def _track_properties(properties, save_images=False):
    """Helping function. Writes track properties to a file."""

    file = open('track_props.txt', 'w')
    file.write('object,minor_axis,major_axis,mean_gray\n')

    for prop in properties:
        fig, ax = plt.subplots()
        img_cont = ax.contour(prop.intensity_image, colors='w')
        ax.clabel(img_cont, fmt='%i', fontsize=12, colors='w')
        ax.contourf(prop.intensity_image, cmap='magma')

        line_file = str(prop.label) + ',' + \
            str(prop.minor_axis_length) + ',' + \
            str(prop.major_axis_length) + ',' + \
            str(prop.mean_intensity) + '\n'

        file.write(line_file)

        if save_images:
            img_filename = 'track_' + str(prop.label) + 
            plt.savefig(filename=img_filename, bbox_inches='tight')
        else:
            plt.show()

    file.close()

    return None
