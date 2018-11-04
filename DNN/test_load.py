#import numpy as np
import cv2
import json
import load_data as ld

labelSet = []

with open('dataset.json') as f:
    data = json.load(f)
    for img in data:
        for region in img["regions"]:
            if region["tagName"] not in labelSet:
                labelSet.append(region["tagName"])


print(labelSet)
X, Y = ld.load_data('dataset.json', labelSet)

print(Y[0])
print(X[0].shape)
print(X[1].shape)
cv2.imshow('image', X[0])

cv2.waitKey(0)
cv2.destroyAllWindows()
