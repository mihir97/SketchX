import cv2
import numpy as np
import sys

def get_rect_rank(rect,h_max,w_max):
    x_mean=(rect[0]+rect[2])/2
    y_mean=(rect[1]+rect[3])/2
    rank = (y_mean/(h_max/6))*w_max+x_mean
    return rank


image = cv2.imread(sys.argv[1])
gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)    #Convert image to grayscale , 0-255

height, width = gray_image.shape[:2]
print(height * width * .005)
img = cv2.adaptiveThreshold(gray_image,255,cv2.ADAPTIVE_THRESH_MEAN_C,cv2.THRESH_BINARY_INV,7,10)   #Adaptive thresholding, geberates bianry image
cv2.imwrite("grayscale.jpg",img)

kernel = np.ones((5,7), np.uint8)       # vary the two parameters for horizontal/vertical grouping
img = cv2.dilate(img, kernel, iterations=4) #iterations is another parameter to vary
cv2.imwrite("dilated.jpg",img)

image, contours, hierarchy = cv2.findContours(img,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)   #Contours generation
contour_rects = []
for (i, j) in zip(contours, hierarchy[0]):
    x,y,w,h = cv2.boundingRect(i)
    if cv2.contourArea(i) > height * width * .0025 and w < width * .9:
        contour_rects.append([x,y,x+w,y+h])

contour_rects.sort(key=lambda x:get_rect_rank(x,height,width))       #Sort contours left to right (Useful for generating html code)

print(contour_rects)
k=0
for x,y,x_w,y_h in contour_rects:
        gray_image = cv2.rectangle(gray_image,(x,y),(x_w,y_h),(0,0,0),2)        #Draw contours
        k += 1
        cv2.putText(gray_image,str(k),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)    #Number the contours
cv2.imwrite("contours.jpg",gray_image)
