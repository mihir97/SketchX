import json
import numpy as np
import cv2

def resize_img(img):
    scale_percent = 25 # percent of original size
    width = int(img.shape[1] * scale_percent / 100)
    height = int(img.shape[0] * scale_percent / 100)
    dim = (width, height)
    # resize image
    resized = cv2.resize(img, None, fx = 0.1, fy = 0.1)
    return resized
 
#Need to rewrite this function
def pad_img(img):
    #Padding an image
    result = np.zeros((64, 128))
    result[:img.shape[0], :img.shape[1]] = img
    #result = cv2.resize(img, (128, 64))
    return result

def generate_test(X, Y):
    #Split data in test and train
    return x, y, x_test , y_test

def load_data(jsonfile, labelSet):
    X = []
    Y = []
    val = False
    with open(jsonfile) as f:
        data = json.load(f)
        for image in data:
            #print(img)
            scale = 0.05
            scale = 1
            height = int(image["height"]*scale)
            width = int(image["width"]*scale)
            name = "images/"+image["id"]+".png"
            img = cv2.imread(name, 0)
            img = cv2.resize(img, (width, height))
            for region in image["regions"]:
                left = region["left"]
                top = region["top"]
                h = region["height"]
                w = region["width"]
                tag = img[int(top*height):int((top+h)*height), int(left*width):int((left+w)*width)]
                if tag.shape[0] != 0 and tag.shape[1] != 0:
                    resized =   cv2.resize(tag, dsize=(128, 64))
                    #X.append(pad_img(tag))
                    #X.append(pad_img(resized))
                    X.append(resized)
                    #X.append(pad_img(resized))
                    if(val):
                        cv2.imshow('tag', resized)
                        cv2.imshow('image', tag)
                        val = False
                    y = np.zeros(len(labelSet))
                    y[labelSet.index(region["tagName"])] = 1
                    Y.append(y)
                else:
                    #print(int(left*width),int(top*height), int((left+w)*width),int((top+h)*height))
                    cv2.imshow('image',tag)
                    return

    #return generate_test(X,Y)
    return np.array(X), Y
