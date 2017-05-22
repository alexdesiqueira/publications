'''



'''

from itertools import product

import numpy as np


def visual_comparison(image, img_groundtruth):
    '''
    '''

    rows, cols = image.shape
    all_px = rows*cols
    img_comp = np.zeros((rows, cols, 3))
    img_px = np.zeros((all_px, 3))
    true_pos, true_neg, false_pos, false_neg = 0, 0, 0, 0

    for i, j in product(range(rows), range(cols)):
        # True Positives
        if (image[i, j], img_groundtruth[i, j]) == (True, True):
            img_comp[i, j, 1] = 1  # TP receives green
            true_pos += 1
        # False Positives
        elif (image[i, j], img_groundtruth[i, j]) == (True, False):
            img_comp[i, j, 0] = 1  # FP receives red
            false_pos += 1
        # False Negatives
        elif (image[i, j], img_groundtruth[i, j]) == (False, True):
            img_comp[i, j] = (1, 1, 0.2)  # FN receives yellow
            false_neg += 1
        # True Negatives
        else:
            img_comp[i, j, 2] = 1  # TN receives blue
            true_neg += 1

    # Putting TP + TN, FP and FN pixels in order.
    # TP: green
    first = true_pos
    img_px[0:first, 1] = True

    # TN: blue
    second = first + true_neg
    img_px[first+1:second, 2] = 1

    # FP: red
    third = second + false_pos
    img_px[second+1:third, 0] = 1

    # FN: yellow
    fourth = third + false_neg
    img_px[third+1:fourth] = (1, 1, 0.2)
    img_px = np.reshape(img_px, newshape=(rows, cols, 3))

    return img_comp, img_px, [true_pos, true_neg, false_pos, false_neg]
