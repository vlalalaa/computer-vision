import numpy as np
import cv2

image1 = cv2.imread('images/img2.jpg')
image1 = cv2.resize(image1, (image1.shape[1] // 2, image1.shape[0] // 2))
print(image1.shape)
cv2.rectangle(image1,(50,80),(320,350),(0, 0, 255), 2)
cv2.putText(image1, "Polishchuk Vlada", (80, 400), cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, (0, 0, 255), 1)

cv2.imshow("Image", image1)


cv2.waitKey(0)
cv2.destroyAllWindows()