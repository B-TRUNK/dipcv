from skimage import io ,img_as_float
import numpy as np
from matplotlib import pyplot as plt


img = io.imread('../../raw_imgs/hq.jpg')
print(img)  #print the numpy array
print(img.min() ,img.max())

#creating a random image using numpy
"""
random_image = np.random.random([500 ,500])
plt.imshow(random_image)
print(random_image)
print(random_image.min() ,random_image.max())

"""

floating_image = img_as_float(img)
print(floating_image.min() ,floating_image.max())
plt.imshow(floating_image)

#multiplying numpy array images
#1 - darkening an image
dark_image = floating_image*0.5
plt.imshow(dark_image)

img[10:200 ,10:200 ,:] = [255 ,0 ,0]
plt.imshow(img)

plt.show()