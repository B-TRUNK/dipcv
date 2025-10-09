from skimage import io
from skimage import restoration #convolution
from matplotlib import pyplot as plt
import numpy as np

img = io.imread('../../raw_imgs/hq.jpg' ,as_gray = True)


psf = np.ones((3 ,3)) / 9
deconvolved, _ = restoration.unsupervised_wiener(img ,psf)

plt.imsave('../../gen_imgs/deconvolved.jpg' ,deconvolved ,cmap = ('gray'))

plt.imshow(deconvolved)

plt.show()