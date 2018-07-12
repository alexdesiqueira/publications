from itertools import product
from math import atan2
from operator import itemgetter
from scipy.ndimage import binary_fill_holes
from skimage.color import gray2rgb
from skimage.draw import line
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from skimage.morphology import skeletonize_3d
from skimage.segmentation import clear_border
from skimage.util import img_as_ubyte

import matplotlib.pyplot as plt
import numpy as np
import timeit


def angles_in_route(refer_pt, points):
    """Returns the angles between the reference point and each point on
    a set.

    Parameters
    ----------
    refer_pt : list
        The point to be used as reference when calculating the angles.
    points : list
        A list of points to calculate the angles .

    Returns
    -------
    angles : list
        List containing the angles between reference_pt and each point
        of the points list.
    """

    angles = []

    for point in points:
        d_row = refer_pt[0] - point[0]
        d_col = refer_pt[1] - point[1]
        angles.append(atan2(d_col, d_row))

    return angles


def clear_rd_border(image):
    """
    """

    aux = np.pad(image, ([1, 0], [1, 0]), mode='constant')
    aux = clear_border(aux)

    return aux[:-2, 1:]


def count_and_save(images, filename='auto_count/auto_images.csv',
                   show_exectime=False):
    """
    """

    aux = count_imagesets(images, show_exectime)

    np.savetxt(filename,
               np.asarray(aux[1], dtype='int'),
               fmt='%i',
               delimiter=',')

    return None


def count_by_region(img_regions):
    """
    """

    # 0: indexes; 1: region tracks; 2: total tracks
    count_auto = [[], [], []]

    for idx, img in enumerate(img_regions[2]):
        aux_trk = []
        count_auto[0].append(idx)

        for idy, reg in enumerate(img):
            trk_area, trk_pts = tracks_classify(reg)
            try:
                aux_trk.append(len(trk_area))
            except TypeError:
                aux_trk.append(1)

        count_auto[1].append(sum(aux_trk))
        count_auto[2].append(aux_trk)

    return count_auto


def count_imagesets(images, show_exectime=False):
    """
    """

    count_sets = [[], []]

    for idx, img in enumerate(images[2]):
        count_sets[0].append(idx)
        if show_exectime:
            start_time = timeit.default_timer()
            sum_tracks, _ = count_regions(img)
            print(timeit.default_timer() - start_time)
        else:
            sum_tracks, _ = count_regions(img)
        count_sets[1].append(sum_tracks)

    return count_sets


def count_regions(regions):
    """
    """

    # 0: indexes; 1: total tracks
    aux_trk, count_reg = [], []

    for idx, reg in enumerate(regions):
        trk_area, trk_pts = tracks_classify(reg)

        try:
            count_reg.append(len(trk_area))
        except TypeError:
            count_reg.append(1)

    sum_tracks = sum(count_reg)

    return sum_tracks, count_reg


def manual_counting(mineral='apatite'):
    """
    """

    if mineral == 'apatite':
        count_manual = [113, 111, 104, 90, 79, 107, 98, 104, 111, 96,
                        80, 125, 104, 110, 84, 112, 92, 106, 103, 122,
                        106, 92, 97, 102, 117, 98, 101, 117, 105, 128]

    elif mineral == 'mica':
        count_manual = [54, 53, 70, 55, 78, 50, 53, 57, 48, 51, 63, 46, 
                        51, 69, 58, 60, 58, 73, 66, 63, 50, 55, 60, 56,
                        40, 60, 69, 52, 59, 74, 51, 48, 61, 56, 47, 72,
                        60, 54, 54, 57, 59, 68, 64, 48, 45, 50, 66, 63,
                        64]

    return count_manual


def pixels_and_neighbors(image):
    """Returns true pixels in a binary image, and their neighbors.

    Parameters
    ----------
    image : (rows, cols) ndarray
        Binary input image.

    Returns
    -------
    px_and_neigh : list
        List containing true pixels (first coordinate) and their true
        neighbors (second coordinate).
    """

    true_rows, true_cols = np.where(image)
    true_pixels, px_and_neigh = [], []

    for i, _ in enumerate(true_cols):
        true_pixels.append([true_rows[i], true_cols[i]])

    for pixel in true_pixels:
        aux = []
        possible_neigh = [[pixel[0]-1, pixel[1]-1],
                          [pixel[0]-1, pixel[1]],
                          [pixel[0]-1, pixel[1]+1],
                          [pixel[0], pixel[1]-1],
                          [pixel[0], pixel[1]+1],
                          [pixel[0]+1, pixel[1]-1],
                          [pixel[0]+1, pixel[1]],
                          [pixel[0]+1, pixel[1]+1]]

        for point in possible_neigh:
            if point in true_pixels:
                aux.append(point)

        px_and_neigh.append([pixel, aux])

    return px_and_neigh


def pixels_interest(image):
    """
    """

    px_neighbors = pixels_and_neighbors(image)
    px_extreme, px_intersect = [[] for i in range(2)]

    for pixel in px_neighbors:
        if len(pixel[1]) == 1:
            px_extreme.append(pixel[0])
        if len(pixel[1]) > 2:
            px_intersect.append(pixel[0])

    return px_extreme, px_intersect


def plot_and_count(image_bin, intensity_image=None, ax=None):

    if ax is None:
        ax = plt.gca()

    total_tracks = 0
    props = regionprops(label(image_bin))

    img_skel = skeletonize_3d(image_bin)
    rows, cols = np.where(img_skel != 0)

    if intensity_image is not None:
        img_rgb = gray2rgb(img_as_ubyte(intensity_image))
    else:
        img_rgb = gray2rgb(img_as_ubyte(image_bin))

    img_rgb[rows, cols] = [255, 0, 255]

    for prop in props:
        obj_info = []
        aux = skeletonize_3d(prop.image)
        trk_area, trk_px = tracks_classify(aux)
        count_auto = count_by_region(regions_and_skel(prop.image))

        total_tracks += np.sum(count_auto[2])

        x_min, y_min, x_max, y_max = prop.bbox
        obj_info.append([prop.centroid[0],
                         prop.centroid[1],
                         str(count_auto[2][0][0])])
        for obj in obj_info:
            ax.text(obj[1], obj[0], obj[2], family='monospace',
                    color='yellow', size='medium', weight='bold')

        if trk_area is not None:
            for px in trk_px:
                route, _ = route_through_array(~aux, px[0], px[1])

                for rx in route:
                    ax.scatter(y_min+rx[1], x_min+rx[0], s=20, c='b')
                for p in px:
                    ax.scatter(y_min+p[1], x_min+p[0], s=20, c='g')

                rows, cols = line(x_min+px[0][0], y_min+px[0][1],
                                  x_min+px[1][0], y_min+px[1][1])
                img_rgb[rows, cols] = [0, 255, 0]

    ax.imshow(img_rgb, cmap='gray')

    return ax, total_tracks


def regions_and_skel(image_bin):
    """
    """

    image_reg = [[], [], []]

    if type(image_bin) is list:
        for idx, img in enumerate(image_bin):
            aux = _aux_regskel(img)

            image_reg[0].append(idx)
            image_reg[1].append(aux[0])
            image_reg[2].append(aux[1])

    elif len(image_bin.shape) == 2:
        aux = _aux_regskel(image_bin)

        image_reg[0].append(0)
        image_reg[1].append(aux[0])
        image_reg[2].append(aux[1])

    return image_reg


def _aux_regskel(image):
    aux_reg, aux_skel = [], []
    img_props = regionprops(label(image))

    for prop in img_props:
        aux_reg.append(prop.image)
        aux_skel.append(skeletonize_3d(prop.image))

    return aux_reg, aux_skel


def track_parameters(first_px, last_px, route):
    """
    """

    dist_px = np.linalg.norm(np.array(first_px) - np.array(last_px))
    dist_diff = np.abs(dist_px-len(route))

    angles = angles_in_route(first_px, route)
    d_row = first_px[0] - last_px[0]
    d_col = first_px[1] - last_px[1]
    angle_px = atan2(d_col, d_row)

    return (dist_px, dist_diff), angle_px, angles


def tracks_classify(image):
    """
    """

    px_ext, px_int = pixels_interest(image)
    trk_areas, trk_points = [], []

    # first, checking the extreme pixels.
    if len(px_ext) == 2:  # only one track (two extreme points).
        return None, None

    while len(px_ext) != 0:
        areas, points = [], []
        if len(px_ext) >= 2:
            for i, j in product(px_ext, px_ext):
                if ((j, i) not in points) and (i is not j):
                    route, _ = route_through_array(~image, i, j)
                    points.append([i, j])
                    areas.append(tracks_by_area(i, j,
                                                route,
                                                image.shape))

        else:
            # then, checking extreme vs. intersection pixels.
            for i, j in product(px_ext, px_int):
                if ((j, i) not in points) and (i is not j):
                    route, _ = route_through_array(~image, i, j)
                    points.append([i, j])
                    areas.append(tracks_by_area(i, j,
                                                route,
                                                image.shape))

            # check if an extreme pixel united with an intersection pixel
            # has another part (constituted by an inter px + an ext px)

        # get the best candidate, erase the points and repeat.
        min_idx = min(enumerate(areas), key=itemgetter(1))[0]

        trk_areas.append(areas[min_idx])
        trk_points.append(points[min_idx])

        for point in points[min_idx]:
            if point in px_ext:
                px_ext.remove(point)

    return trk_areas, trk_points


def tracks_by_area(pt_a, pt_b, route, img_shape):
    """
    """

    sum_area = 0

    # first, we plot the Euclidean line between the two points.

    region_area = np.zeros(img_shape)
    rows, cols = line(pt_a[0], pt_a[1],
                      pt_b[0], pt_b[1])
    region_area[rows, cols] = True

    # then, we plot the route.
    for pt in route:
        region_area[pt[0], pt[1]] = True

    # we fill the region and remove small objects.
    region_area = binary_fill_holes(region_area)

    # at the end, we exclude the Euclidean line and the route,
    # to get only the middle. then, we obtain the area within the
    # region.

    region_area[rows, cols] = False

    for pt in route:
        region_area[pt[0], pt[1]] = False

    props = regionprops(label(region_area))

    if props:
        for prop in props:
            sum_area += prop.area
    else:
        sum_area = 0

    return sum_area
