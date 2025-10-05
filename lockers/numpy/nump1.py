import numpy as np
import skimage as ski
import cv2


a = [1 ,2 ,3 ,4 ,5]
print(a*2)
print("Before Using Numpy")



b = np.array(a)
print(b*2)
print("After Using Numpy")

#use numpy to generate an array
zero_matrix = np.zeros((2 ,2))
print("======== \n zero matrix : \n ======== \n" ,zero_matrix)

custom_matrix = np.full((3 ,3) ,4)
print("======== \n custom matrix : \n ======== \n" ,custom_matrix)

identity_matrix = np.eye(3)
print("======== \n identity matrix : \n ======== \n" ,identity_matrix)

random_matrix = np.random.random((2 ,2))
print("======== \n random matrix : \n ======== \n" ,random_matrix)

#=======================================================================
#slicing a matrix
a = np.array( [ [1 ,2 ,3 ,4] ,[5 ,6 ,7 ,8] ,[9 ,10 ,11 ,12] ] )
print(a)

b = a[:2]
print(b)

#========================================================================
#array indexing
print(a[0 ,0] ,a[1,1] ,a [2, 0])

#========================================================================
#MATRIX OPERATIONS

x = np.array( [[1 ,2] ,[3 ,4]] ,dtype = np.float64 )
y = np.array( [[5 ,6] ,[7 ,8]] ,dtype = np.float64 )

print("======== \n Matrices_MUL : \n ======== \n" ,np.multiply(x ,y))
print("======== \n Matrices_SUM : \n ======== \n" ,np.add(x ,y))
print("======== \n Matrices_SUM-at a spicific axis : \n ======== \n" ,np.sum(x ,axis=0))
print("======== \n Matrix-Transpose : \n ======== \n" ,x.T)

#=========================================================================
#Real Operations on an Image
img = cv2.imread("../../raw_imgs/hq.jpg" ,1)

cv2.imshow("pic" ,img)
print(img)

tinted_img = (img * [1. ,0. ,0.])
cv2.imshow("tinted" ,tinted_img)
print(tinted_img)
#cv2.imwrite("../../raw_imgs/hq-tinted-b.jpg" ,tinted_img)


#image_rotation
rotated_img = cv2.rotate(img ,cv2.ROTATE_90_CLOCKWISE)
cv2.imwrite("../../raw_imgs/hqrot.jpg" ,rotated_img)

#image_blurring(gaussian_filter)
blurred_img = cv2.GaussianBlur(img ,(7 ,7) ,0)
cv2.imwrite("../../raw_imgs/hqblur.jpg" ,blurred_img)

#image_blurring(gaussian_filter)
denoised_img = cv2.fastNlMeansDenoisingColored(
    img, None, h=10, hColor=10, templateWindowSize=7, searchWindowSize=21
)
cv2.imwrite("../../raw_imgs/hqdenoised.jpg" ,denoised_img)

cv2.waitKey(0)
cv2.destroyAllWindows()
