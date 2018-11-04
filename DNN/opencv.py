import numpy as np
import cv2
import json

width = 900
height = 1200

f = open('dataset.json')

data = json.load(f)
js = None
for img in data:
    if img["id"] == "1b1f4c8c-92a7-4000-8f07-471d6f8016b1":
        js = img
img = cv2.imread('images/1b1f4c8c-92a7-4000-8f07-471d6f8016b1.png',0)

width = js["width"]
height = js["height"]

height, width = img.shape

for r in js["regions"]:
    left = r["left"]
    top = r["top"]
    w = r["width"]
    h = r["height"]    
    tag = img[int(top*height):int((top+h)*height), int(left*width):int((left+w)*width)]
    cv2.imshow(r["tagName"], tag)
    #cv2.rectangle(img,(int(left*width),int(top*height)),(int((left+w)*width),int((top+h)*height)),(0,255,0),3)
    
#img = cv2.resize(img, None, fx = 0.5, fy = 0.5)
#cv2.imshow('image', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

