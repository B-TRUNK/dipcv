from skimage import io
from skimage import restoration #convolution
from matplotlib import pyplot as plt
import numpy as np
import scipy.stats as st


img = io.imread('../../raw_imgs/hq.jpg' ,as_gray = True)

#psf = np.ones((3 ,3)) / 9

def gkern(kernlen=21, nsig=2):  # Returns a 2D Gaussian kernel.
    lim = kernlen / 2 + (kernlen % 2) / 2
    x = np.linspace(-lim, lim, kernlen + 1)
    kern1d = np.diff(st.norm.cdf(x))
    kern2d = np.outer(kern1d, kern1d)
    return kern2d / kern2d.sum()

psf = gkern(5, 3)  # Kernel length and sigma
print(psf)


deconvolved, _ = restoration.unsupervised_wiener(img ,psf)

plt.imsave('../../gen_imgs/deconvolved2.jpg' ,deconvolved ,cmap = ('gray'))

plt.imshow(deconvolved)

plt.show()