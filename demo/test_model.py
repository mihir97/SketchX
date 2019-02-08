import tflearn
import numpy as np
import cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression
import sys
import region



labelSet = ['CheckBox', 'Button', 'Label', 'RadioButton', 'TextBox', 'Heading', 'ComboBox', 'Link', 'Image', 'Paragraph']
threshold = 0.0    

def predict_from_model(rects, img):
    #labelSet = ['CheckBox', 'Button', 'Label', 'RadioButton', 'TextBox', 'Heading', 'ComboBox', 'Link', 'Image', 'Paragraph']
    convnet = input_data(shape= [None, 64, 128, 1], name='input')
    convnet = conv_2d(convnet, 64, 4, activation='relu')
    convnet = max_pool_2d(convnet, 4)

    convnet = conv_2d(convnet, 32, 4, activation='relu')
    convnet = max_pool_2d(convnet, 4)

    convnet = fully_connected(convnet, 32, activation='relu')
    convnet = dropout(convnet, 0.98)

    convnet = fully_connected(convnet, len(labelSet), activation = 'softmax')
    convnet = regression(convnet, optimizer='adam', learning_rate=0.001, loss='categorical_crossentropy', name='targets')

    model = tflearn.DNN(convnet)

    model.load('tflearncnn3.model')

    #img = cv2.imread(sys.argv[1], 0)
    Xs = []
    for i, rect in enumerate(rects):
        x, y, w, h = rect
        tag = img[int(y):int((y+h)), int(x):int((x+w))]
        #cv2.imwrite('timepass'+ str(i) +'.jpg', tag)
        resized =   cv2.resize(tag, dsize=(128, 64))
        Xs.append(resized)
                
    Xs = np.array(Xs)
    print(Xs.shape)
    Xs = Xs.reshape([-1,64, 128, 1])
    num_bufs = int(len(Xs)/50) + 1
    Ys = []*len(Xs)
    for i in range(num_bufs):
        if (i+1)*50 > len(Xs):
            Ys[i*50:] = model.predict(Xs[i*50:])
        else:
            Ys[i*50:(i+1)*50] = model.predict(Xs[i*50:(i+1)*50])
    return Ys
    
if __name__ == '__main__':
    img = cv2.imread(sys.argv[1])
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)    #Convert image to grayscale , 0-255

    rects = region.get_regions(img)
    
    #resized = cv2.resize(img, dsize=(128, 64))

    #resized = np.array(resized)

    #resized = resized.reshape([1,64, 128, 1])

    

    Y = predict_from_model(rects, gray_image)                    
    #y = labelSet[np.argmax(Y[0])]
    for i, rect in enumerate(rects):
        x,y,w,h  = rect
        conf = np.amax(Y[i])
        if  conf > threshold:
            img = cv2.rectangle(img,(x,y),(x+w,y+h),(0,0,0),2)        #Draw contours
            cv2.putText(img,labelSet[np.argmax(Y[i])] + ':' + str(conf),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)    #Number the contours
    cv2.imwrite("output.jpg",img)
    
    #Y = labelSet[np.argmax(y[0])]
    
