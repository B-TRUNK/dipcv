import cv2
from skimage.filters import sobel

img = cv2.imread("raw_imgs/hq.jpg" ,1)
img2 = sobel(img) 


cv2.imshow("pic" ,img)
print(img.shape)

cv2.waitKey(0)
cv2.destroyAllWindows()
