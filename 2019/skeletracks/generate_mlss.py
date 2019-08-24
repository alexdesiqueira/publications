import os
import shutil

from glob import glob
from skimage import io
from skimage.restoration import denoise_tv_chambolle
from subprocess import Popen

WEIGHT_FILTER = 0.05


# checking if the folder 'figures' exists.
if not os.path.isdir('./filt_figures'):
    os.mkdir('./filt_figures')

# getting all image filenames.
filenames = glob('orig_figures/*.tif')

# generating denoised images with nice names.
for idx, filename in enumerate(filenames):
    image = io.imread(filename, as_gray=True)
    image = denoise_tv_chambolle(image, weight=WEIGHT_FILTER)
    filt_fname = 'filt_figures/' + filename.split('/')[1][:-4] + '.png'
    io.imsave(fname=filt_fname, arr=image)

# executing Jansen-MIDAS.
try:
    proc = Popen(['octave', 'obtain_mlss.m'],
                 cwd='./jansenmidas')
    proc.wait()
except:
    raise

# removing denoised images.
shutil.rmtree('./filt_figures')
