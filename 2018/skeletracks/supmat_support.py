from itertools import product
from math import atan2
from operator import itemgetter
from scipy.ndimage import binary_fill_holes
from skimage.color import gray2rgb
from skimage.draw import line
from skimage.graph import route_through_array
from skimage.measure import label, regionprops
from skimage.morphology import remove_small_objects

import matplotlib.pyplot as plt
import numpy as np
import statistics as stats


def angles_between_points(points):
    """Returns the angles between each point and the previous one on a
    set.

    Parameters
    ----------
    points : list
        A list of points.

    Returns
    -------
    angles : list
        List containing the angles between each point and the previous
        one.
    """

    angles = []

    for idx, point in enumerate(points):
        d_row = point[0] - points[idx-1][0]
        d_col = point[1] - points[idx-1][1]
        angles.append(atan2(d_col, d_row))

    return angles


def plot_angles(angles):
    """
    """

    rows = len(angles)
    x_angles = np.arange(0, rows)
    y_zeros = np.zeros(rows)
    y_pi = np.ones(rows) * np.pi

    plt.figure()
    plt.plot(angles, 'k.')
    plt.vlines(x_angles, [0], angles, 'm')

    plt.plot(x_angles, y_zeros, 'k--',
             x_angles, -y_pi, 'k--',
             x_angles, y_pi, 'k--' )

    plt.axis([0, rows, -4, 4])
    plt.show()

    return None


def plot_tracks(image, trk_averages, trk_points):
    '''
    '''

    if trk_averages is None:
        print('No averages (possibly only one track).')
        return None

    image_rgb = gray2rgb(image)

    # preparing the plot window.
    fig, ax = plt.subplots(nrows=1, ncols=1)

    label_text = ''

    for trk_idx, trk_pt in enumerate(trk_points):

        # calculating route and distances.
        route, _ = route_through_array(~image, trk_pt[0], trk_pt[1])
        distances, _, _ = track_parameters(trk_pt[0], trk_pt[1], route)

        # generating minimal distance line.
        rows, cols = line(trk_pt[0][0], trk_pt[0][1],
                          trk_pt[1][0], trk_pt[1][1])
        image_rgb[rows, cols] = [0, 255, 0]

        # plotting minimal distance and route.
        ax.imshow(image_rgb, cmap='gray')

        for rt_pt in route:
            ax.scatter(rt_pt[1], rt_pt[0], c='b')

        #plotting extreme points.
        for pt in trk_pt:
            ax.scatter(pt[1], pt[0], c='g')

        # preparing the label text.
        label_text += 'Track index: ' + str(trk_idx) + \
                      '\n Extreme points: ' + str(trk_pt[0]) + ', '  + str(trk_pt[1]) + \
                      '\n Length (route): ' + str(len(route)) + \
                      '\n Euclidean distance: ' + str(distances[0]) + \
                      '\n Difference: ' + str(distances[1]) + '\n\n'

    # removing plot ticks.
    ax.axes.get_xaxis().set_ticks([])
    ax.axes.get_yaxis().set_ticks([])

    # removing plot 'spines'.
    ax.spines['top'].set_visible(False)
    ax.spines['bottom'].set_visible(False)
    ax.spines['left'].set_visible(False)
    ax.spines['right'].set_visible(False)

    # setting x label.
    ax.set_xlabel(label_text)

    return None


def plot_auxiliar(var_human, var_algo, algo_color='b', ax=None):
    '''
    '''

    if ax is None:
        ax = plt.gca()

    rows = len(var_human)
    x_trk = np.arange(0, rows)
    # human counting.
    ax.plot(var_human, '.', color='0.60')
    ax.vlines(x_trk, [0], var_human, color='0.60')
    # algorithm counting.
    ax.plot(var_algo, '.', color=algo_color)
    ax.vlines(x_trk, [0], var_algo, color=algo_color)

    ax.axis([-1, rows, 0, 700])
    ax.get_xaxis().set_ticks([])
    ax.yaxis.set_major_locator(MultipleLocator(150))

    ax.set_xlabel('Human:\n' \
                  '* Sample st dev: ' + str(np.round(stats.stdev(var_human), decimals=2)) + '\n' \
                  '* Pop st dev: ' + str(np.round(np.std(var_human), decimals=2)) + '\n' \
                  '* Poisson variation: ' + str(np.round(np.sqrt(np.mean(var_human)), decimals=2)) + '\n' \
                  '* SSD / PV: ' + str(np.round(stats.stdev(var_human) / np.sqrt(np.mean(var_human)),
                                                decimals=2)) + '\n\n' \
                  'Algorithm:\n' \
                  '* Sample st dev: ' + str(np.round(stats.stdev(var_algo), decimals=2)) + '\n' \
                  '* Pop st dev: ' + str(np.round(np.std(var_algo), decimals=2)) + '\n' \
                  '* Poisson variation: ' + str(np.round(np.sqrt(np.mean(var_algo)), decimals=2)) + '\n' \
                  '* SSD / PV: ' + str(np.round(stats.stdev(var_algo) / np.sqrt(np.mean(var_algo)),
                                                decimals=2))
                 )
    return ax

