import cv2
import numpy as np
img = np.zeros((500, 400, 3), np.uint8) #перша висота, потім ширина, потім колір
#img[:] = 245, 135, 66 #rgb = bgr

#img[100:150, 200:250] = 245, 135, 66 #знову спочатку висота, а потім ширина у відступах

cv2.rectangle(img, (100, 100), (200, 200), (245, 135, 66), 1)  #початку координати верхньої лівої точки, потім правої нижньої, потім колір, потім контур
#cv2.rectangle(img, (100, 100), (200, 200), (245, 135, 66), -1)

cv2.line(img, (100, 100), (200, 200), (245, 135, 66), 1) #спочатку ліва точка, потім права

print(img.shape)
cv2.line(img, (img.shape[1] // 2, 0), (img.shape[1]//2, img.shape[0]), (245, 135, 66), 1)
cv2.line(img, (0, img.shape[0] // 2), (img.shape[1], img.shape[0] // 2), (245, 135, 66), 1)

cv2.circle(img, (200, 200), 30, (245, 135, 66), -1)
cv2.putText(img, "Polishchuk Vlada", (230, 150), cv2.FONT_HERSHEY_PLAIN,1, (245, 135, 66))


cv2.imshow("image", img)
cv2.waitKey(0)
cv2.destroyAllWindows()