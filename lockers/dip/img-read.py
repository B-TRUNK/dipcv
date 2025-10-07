#1 - pillow lib:for image manip. and processing lib 
# ,used for cropping ,resizing ,basic filtering ,but for higher processing 
# ,other libs like cv ,skimage ,sklearn are used

from PIL import Image
import numpy as np
from matplotlib import pyplot as plt ,image as mpimg
from skimage import io ,img_as_float ,img_as_ubyte
import cv2
"""
#note images read by PIL are not arrays
img = Image.open('../../raw_imgs/hq.jpg')
print(img)
img.show()
print(img.format)

#converting into an array image 
numpy_image = np.array(img)
print(numpy_image)
"""
print('====================================================')
"""
#2 - Matplotlib
img = mpimg.imread('../../raw_imgs/hq.jpg')
print('Matplot image reading data :' ,type(img))
print('Image Shape :' ,img.shape)

plt.imshow(img)
plt.colorbar()
"""


print('====================================================')
"""
#3 - Skimage
img = io.imread('../../raw_imgs/hq.jpg')
print('Skimage image reading data :' ,type(img))
plt.imshow(img)
"""

print('====================================================')
#4 - OpenCV

grey_image = cv2.imread('../../raw_imgs/hq.jpg' ,0)
cv2.imshow('Grey_Image' ,grey_image)

colorful_image = cv2.imread('../../raw_imgs/hq.jpg' ,1)
cv2.imshow('Colorful_Image' ,colorful_image)


cv2.waitKey(3000)
cv2.destroyAllWindows()


plt.show()