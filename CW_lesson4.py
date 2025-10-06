import cv2
import numpy as np
img = cv2.imread('images/img6.jpg')
scale = 4
img = cv2.resize(img, (612, 408))
print(img.shape)

img_copy = img.copy()

img_copy = cv2.cvtColor(img_copy, cv2.COLOR_BGR2GRAY)
img_copy = cv2.GaussianBlur(img_copy, (5, 5), 2) #розмиваємо для того щоб оптимізувати програму

img_copy = cv2.equalizeHist(img_copy) #посилення контрасту
img_copy = cv2.Canny(img_copy, 100, 150)


#Пошук контурів
contours, hierarchy = cv2.findContours(img_copy, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE) #RETR_EXTERNAL отримує зовнішній контур (саму форму людини, а не її очі, губи, тощо),  а CHAIN_APPROX - апроксимація (процес наближеного вираження одних величин або об'єктів через інші)
img_copy_color = img.copy()
#малювання контурів, прямокутників та тексту
for  cnt in contours:
    area = cv2.contourArea(cnt) #визначаємо площу контура, повертає площу в пікселях
    if area > 100:
        x, y, w, h = cv2.boundingRect(cnt) #повертаємо найменший прямокутник який повністю в собі містить контур
        #малюємо контур
        cv2.drawContours(img_copy_color, [cnt], -1, (0, 255, 0), 2) #-1 малюємо всі контури масиву, далі колір і товщина
        cv2.rectangle(img_copy_color, (x, y), (x + w, y + h), (0, 255, 0), 2)
        text_y = y - 5 if y-5 > 10 else y + 15
        text = f"x: {x}, y: {y}, S:{int(area)}"
        cv2.putText(img_copy_color, text, (x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 1)






cv2.imshow('Borders', img_copy)
cv2.imshow('Copy border', img_copy_color)
cv2.waitKey()
cv2.destroyAllWindows()