#this code is to test functionality in skimage lib
from skimage import io
from skimage.transform import rescale ,resize ,downscale_local_mean
from skimage.filters import roberts ,sobel ,scharr ,prewitt
from skimage.feature import canny
from matplotlib import pyplot as plt

img = io.imread('../../raw_imgs/hq.jpg' ,as_gray = True)


#1 - Basic Processing
"""
rescaled_img    = rescale(img ,1.0/4.0 ,anti_aliasing = True)
resized_img     = resize(img ,(200 ,200))
downsized_img   = downscale_local_mean(img ,(4 ,3))

plt.imshow(rescaled_img)
plt.imshow(resized_img)
plt.imshow(downsized_img)
"""

#2 - Edge Detection

edge_roberts = roberts(img)
plt.imshow(edge_roberts ,cmap = 'Blues')
edge_sobel = sobel(img)
edge_scharr = scharr(img)
edge_prewitt = prewitt(img)
edge_canny = canny(img ,sigma = 1) #the more sigma the less pixels shown

fig, axes = plt.subplots(nrows=2, ncols=2, sharex=True, sharey=True, figsize=(8, 8))
ax = axes.ravel()

ax[0].imshow(img, cmap=plt.cm.gray)
ax[0].set_title('Original image')

ax[1].imshow(edge_roberts, cmap=plt.cm.gray)
ax[1].set_title('Roberts Edge Detection')

ax[2].imshow(edge_sobel, cmap=plt.cm.gray)
ax[2].set_title('Sobel')

ax[3].imshow(edge_scharr, cmap=plt.cm.gray)
ax[3].set_title('Scharr')

for a in ax:
    a.axis('off')

plt.tight_layout()

plt.imshow(edge_canny)

plt.show()