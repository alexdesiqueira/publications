


def jansenmidas():

    return None

def confusionmatrix():

    return None

def matthewscc():

    return None

from scipy import ndimage
from scipy import signal
from skimage.color import rgb2gray
import numpy as np


def atrous_algorithm(vector, level=3):
    '''
    Applies i levels of the a trous algorithm on the vector vec.
    '''
    if i == 0:
        output = np.copy(vec)
    else:
        m = vec.size
        output = np.zeros(m+(2**i-1)*(m-1))
        # zeroes array depends on vector size and starlet level
        k = 0
        for j in range(0, m+(2**i-1)*(m-1), (2**i-1)+1):
            output[j] = vec[k]
            k += 1

    # normalization
    output = np.atleast_2d(output)  # 2D vectors requires less effort

    return output


def mlssaux(image, detail, initial_level, level):
    '''
    '''
    sum_aux = 0

    for i in range(initial_level, level):
        ratio = 255/(detail[i].max())
        temp = np.array(ratio*detail[i], np.uint8)
        sum_aux = sum_aux + temp

    aux = np.array(sum_aux, dtype=np.uint8)
    # possivelmente normalizar aqui

    output = np.array((aux != 0), dtype=np.uint8)*255

    return output


def starlet(image, L=6):
    '''
    Applies L levels of the starlet wavelet transform on the image image.
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

    # resulting vectors
    m, n = image.shape
    approx = np.empty([L, m, n], float)
    detail = np.empty([L, m, n], float)

    h1D_filter = np.array([1, 4, 6, 4, 1])*(1./16)

    # mirroring parameter: lower pixel number
    if m > n:
        par = n
    else:
        par = m

    aux_approx = np.pad(image, (par, par), 'symmetric')

    # starlet application
    for level in range(L):
        prev_image = aux_approx
        h2D_filter = atrousalg(h1D_filter, level)

        ''' approximation and detail coefficients '''
        aux_approx = signal.fftconvolve(prev_image, h2D_filter,
                                        mode='same')
        aux_detail = prev_image - aux_approx

        ''' mirroring correction '''
        approx[level] = aux_approx[par:m+par, par:n+par]
        detail[level] = aux_detail[par:m+par, par:n+par]

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
        print('Level: %s' % str(level))
        result[level] = mlssaux(image,
                                detail[0:level],
                                initial_level,
                                level)

    return detail, result
