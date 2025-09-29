import cv2
import numpy as np

image1 = cv2.imread('images/img2.jpg')
image1 = cv2.resize(image1, (image1.shape[1] // 4, image1.shape[0] // 4))
image1 = cv2.cvtColor(image1, cv2.COLOR_BGR2GRAY)
image1 = cv2.Canny(image1, 120, 120)
kernel = np.ones((3, 3), np.uint8)
image1 = cv2.dilate(image1, kernel, iterations=1)
image1 = cv2.erode(image1, kernel, iterations=1)


image2 = cv2.imread('images/img3.jpg')
image2 = cv2.resize(image2, (image2.shape[1] // 2, image2.shape[0] // 2))
image2 = cv2.cvtColor(image2, cv2.COLOR_BGR2GRAY)
image2 = cv2.Canny(image2, 400, 400)
kernel = np.ones((2, 2), np.uint8)
image2 = cv2.dilate(image2, kernel, iterations=1)
image2 = cv2.erode(image2, kernel, iterations=1)


cv2.imshow('person', image1)
cv2.imshow('email', image2)

cv2.waitKey(0)
cv2.destroyAllWindows()