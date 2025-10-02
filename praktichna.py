import cv2
import numpy as np
img = np.zeros((400, 600, 3), np.uint8)
img[:] = 247, 195, 183
image = cv2.imread('images/img4.jpg')
image = cv2.resize(image, (120, 120))
img[30:150, 30:150] = image

cv2.rectangle(img, (10, 10), (590, 390), (84, 21, 10), 3)

cv2.putText(img, "Polishchuk Vlada", (180, 90), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 0), 3)
cv2.putText(img, "Computer Vision Student", (180, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (88, 88, 88), 2)

cv2.putText(img, "Email: vladonkapol@gmail.com", (180, 220), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (84, 21, 10))
cv2.putText(img, "Phone: +380687432661", (180, 260), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (84, 21, 10))
cv2.putText(img, "08/12/2009", (180, 300), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (84, 21, 10))

cv2.putText(img, "OpenCV Business Card", (120, 360), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0), 2)

image2 = cv2.imread('images/img5.png')
image2 = cv2.resize(image2, (90, 90))
img[230:320, 470:560] = image2

cv2.imwrite("business_card.png", img)

cv2.imshow("resume", img)
cv2.waitKey(0)
cv2.destroyAllWindows()