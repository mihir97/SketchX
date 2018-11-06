import cv2
import numpy as np
import sys
from sklearn.cluster import KMeans
def get_rect_rank(rect):
    x_mean=(rect[0]+rect[2])/2
    y_mean=(rect[1]+rect[3])/2
    rank = (y_mean/50)*5000+x_mean
    return rank

def find_contour_centers(contours_rect):
    contours_cent = []
    for x,y,x_w,y_h in contour_rects:
        contours_cent.append((x+(x_w - x)/2, 10*(y+(y_h -y)/2)))

    return contours_cent

def combine_rectangles(cluster_label, contours_rect, k):
    clusters = []
    for i in range(k):
        clusters.append([])
    for contour_rect, y in zip(contours_rect, cluster_label):
        #print(contour_rect)
        #print(y)
        clusters[y].append(contour_rect)

    clustered_rect = []
    for cluster in clusters:
        maxes = np.max(cluster, axis=0)
        mins = np.min(cluster, axis = 0)
        #print(maxes)
        #print(mins)
        clustered_rect.append([mins[0], mins[1], maxes[2], maxes[3]])
    return clustered_rect

img = cv2.imread(sys.argv[1],0)
img = cv2.resize(img, (int(img.shape[1]/2), int(img.shape[0]/2)))
img = cv2.medianBlur(img,5)

#(thresh, img) = cv2.threshold(img, 160, 255, cv2.THRESH_BINARY)
img = cv2.adaptiveThreshold(img,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,11,2)

contour_img= img.copy()
word_img= img.copy()

image, contours, hierarchy = cv2.findContours(img,cv2.RETR_CCOMP,cv2.CHAIN_APPROX_SIMPLE)
contour_rects = []
#out_img = cv2.drawContours(out_img, contours, -1, (0,0,0), 3)
k=0
    
for (i, j) in zip(contours, hierarchy[0]):
    if cv2.contourArea(i) > 10 and j[2] == -1 :
        """ Minimum Area Rectangle
        rect = cv2.minAreaRect(i)
        box = cv2.boxPoints(rect)
        box = np.int0(box)
        out_img = cv2.drawContours(out_img,[box],0,(0,0,255),2)
        """
        
        x,y,w,h = cv2.boundingRect(i)
        contour_rects.append([x,y,x+w,y+h])
        
        
#print contour_rects
#contour_rects.sort(key=lambda x:get_rect_rank(x))

Z = find_contour_centers(contour_rects)
#print(Z)
Z = np.array(Z)
#print(Z)
# Number of clusters
K = 8
kmeans = KMeans(n_clusters=K)
# Fitting the input data
kmeans = kmeans.fit(Z)
# Getting the cluster labels
labels = kmeans.predict(Z)
# Centroid values
centroids = kmeans.cluster_centers_
print(centroids)
clustered_rects = combine_rectangles(labels, contour_rects, K)
#print contour_rects
color = [(255,0,0), (0, 0,255), (0,255,0)]
#for [x,y,x_w,y_h], lab in zip(contour_rects, labels):
for x,y,x_w,y_h in clustered_rects:
    #if lab == 0:
        #cv2.imwrite('word_' + str(k).zfill(3) + '.jpg',out_img[y:y+h,x:x+w])
        #contour_img = cv2.rectangle(contour_img,(x,y),(x_w,y_h),color[lab],2)
        contour_img = cv2.rectangle(contour_img,(x,y),(x_w,y_h),(0,0,0),2)
        k += 1
        #cv2.putText(contour_img,str(k),(x,y),cv2.FONT_HERSHEY_SIMPLEX, 1, 2)
#cv2.imwrite("contours.jpg",contour_img)

cv2.imshow('image', contour_img)
cv2.waitKey(0)
cv2.destroyAllWindows()

'''
for x,y,x_w,y_h in contour_rects:
    word = word_img[y:y_h,x:x_w].copy()
    k += 1
    cv2.imwrite('word_' + str(k).zfill(3) + '.jpg',word)
'''
