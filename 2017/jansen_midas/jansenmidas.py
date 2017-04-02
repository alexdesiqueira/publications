


def jansenmidas():

    return None

def confusionmatrix():

    return None

def matthewscc():

    return None

from scipy import ndimage
from scipy import signal
from skimage.color import rgb2gray
from skimage.util import img_as_bool
import numpy as np


def atrous_algorithm(vector, level=3):
    '''
    Applies i levels of the a trous algorithm on vector.
    '''

    if level == 0:
        output = np.copy(vector)
    else:
        rows = vector.size
        output = np.zeros(rows + (2**level - 1) * (rows - 1))
        # zeroes array depends on vector size and starlet level
        k = 0
        for j in range(0,
                       rows + (2**level - 1)*(rows-1),
                       (2**level - 1) + 1):
            output[j] = vector[k]
            k += 1

    # normalization
    output = np.atleast_2d(output)  # 2D vectors require less effort

    return output


def mlss_auxiliar(image, detail, initial_level, level):
    '''
    '''
    sum_details = 0

    for i in range(initial_level, level):
        temp = detail[i] * (image.max()/detail[i].max())
        sum_details += temp

    # possivelmente normalizar aqui

    # output = np.array((sum_aux != 0), dtype=np.uint8)*255

    output = img_as_bool(sum_details)

    return output


def starlet(image, level=6):
    '''
    Applies level levels of the starlet wavelet transform on image.
    '''

    # preliminar vars
    if type(image) is np.ndarray:
        image = np.array(image, float)
    elif type(image) is str:
        try:
            image = ndimage.imread(image, flatten=True)
        except:
            print('Sorry. Data type not understood')
            raise

    # resulting vector
    rows, cols = image.shape
    approx = np.zeros([level, rows, cols])
    detail = np.zeros([level, rows, cols])

    h1D_filter = np.array([1, 4, 6, 4, 1])*(1./16)

    # mirroring parameter: lower pixel number
    if rows > cols:
        param = cols
    else:
        param = rows

    aux_approx = np.pad(image, (param, param), 'symmetric')

    # starlet application
    for level in range(level):
        prev_image = aux_approx
        h2D_filter = atrous_algorithm(h1D_filter, level)

        ''' approximation and detail coefficients '''
        aux_approx = signal.fftconvolve(prev_image, h2D_filter,
                                        mode='same')
        aux_detail = prev_image - aux_approx

        ''' mirroring correction '''
        approx[level] = aux_approx[param:rows+param, param:cols+param]
        detail[level] = aux_detail[param:rows+param, param:cols+param]

    return approx, detail


def mlss(image, initial_level=2, level=6):
    '''
    '''
    print('Starting MLSS...')

    try:
        rows, cols = image.shape
    except:
        print('Sorry. Data type not understood')
        raise
    
    result = np.zeros([level, rows, cols])

    # starlet application
    print('Processing Starlet...')
    __, detail = starlet(image, level)

    # segmentation
    print('Processing segmentation...')
    for level in range(initial_level, level):
        print('level: %s' % str(level))
        result[level] = mlss_auxiliar(image,
                                      detail[0:level],
                                      initial_level,
                                      level)

    return detail, result
