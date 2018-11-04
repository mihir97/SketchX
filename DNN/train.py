import load_data as ld
import json
import tflearn
import cv2
from tflearn.layers.conv import conv_2d, max_pool_2d
from tflearn.layers.core import input_data, dropout, fully_connected
from tflearn.layers.estimator import regression

labelSet = []

with open('dataset.json') as f:
    data = json.load(f)
    for img in data:
        for region in img["regions"]:
            if region["tagName"] not in labelSet:
                labelSet.append(region["tagName"])


print(labelSet)
X, Y = ld.load_data('dataset.json', labelSet)
cv2.imshow('image', X[0])
cv2.waitKey(0)
cv2.destroyAllWindows()

print(X[0].shape)
X = X.reshape([-1, 64, 128,1])
print(X[0].shape)
print(Y[0])
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

model.fit({'input':X}, {'targets':Y}, n_epoch = 25,
          validation_set = 0.2,
          snapshot_step = 500, show_metric= True, run_id = 'mnist')


#just saves weights (not similar to pickle)
model.save('tflearncnn3.model')

'''
model.load('tflearncnn.model')

print(model.predict([test_x[1]]))
'''

