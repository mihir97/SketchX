#!/usr/bin/env python
'''
Usage:
    ./ssearch.py input_image (f|q)
    f=fast, q=quality
Use "l" to display less rects, 'm' to display more rects, "q" to quit.
'''
 
import sys
import cv2
import numpy as np
import test_model as model

def find_rect_centers(contours_rects):
    contours_cent = []
    for x,y,w,h in contours_rects:
        contours_cent.append((x+(w)/2, 10*(y+(h)/2)))

    return contours_cent

def get_rect_rank(rects):
    rect, _ = rects
    x_mean=rect[0]+(rect[2])/2
    y_mean=rect[1]+(rect[3])/2
    rank = (y_mean/50)*5000+x_mean
    return rank

def get_y(rects):
    _, y = rects
    return y

if __name__ == '__main__':
    # If image path and f/q is not passed as command
    # line arguments, quit and display help message
    if len(sys.argv) < 3:
        print(__doc__)
        sys.exit(1)
 
    # speed-up using multithreads
    cv2.setUseOptimized(True);
    cv2.setNumThreads(4);
 
    # read image
    im = cv2.imread(sys.argv[1])
    # resize image
    newHeight = 400
    newWidth = int(im.shape[1]*newHeight/im.shape[0])
    im = cv2.resize(im, (newWidth, newHeight))    
 
    # create Selective Search Segmentation Object using default parameters
    ss = cv2.ximgproc.segmentation.createSelectiveSearchSegmentation()
 
    # set input image on which we will run segmentation
    ss.setBaseImage(im)
 
    # Switch to fast but low recall Selective Search method
    if (sys.argv[2] == 'f'):
        ss.switchToSelectiveSearchFast()
 
    # Switch to high recall but slow Selective Search method
    elif (sys.argv[2] == 'q'):
        ss.switchToSelectiveSearchQuality()
    # if argument is neither f nor q print help message
    else:
        print(__doc__)
        sys.exit(1)
 
    # run selective search segmentation on input image
    rects = ss.process()
    print('Total Number of Region Proposals: {}'.format(len(rects)))
     
    # number of region proposals to show
    numShowRects = 100
    # increment to increase/decrease total number
    # of reason proposals to be shown
    increment = 50

    img = cv2.imread(sys.argv[1], 0)
    # resize image
    #newHeight = 700
    #newWidth = int(img.shape[1]*200/img.shape[0])
    img = cv2.resize(img, (newWidth, newHeight))    
 
    Ys = model.predict_from_model(rects, img)
    #print(Ys)
    shortlist_rects = []
    for rect, y in zip(rects, Ys):
        i = np.argmax(y)
        if y[i] > 0.9:
            #shortlist_rects.append([rect,y])
            shortlist_rects.append([rect,np.max(y)])

    #centers = find_rect_centers(shortlist_rects)

    shortlist_rects.sort(key=lambda x:get_rect_rank(x))

    print("The number of ROIs are:", len(shortlist_rects))
    
    isDone = [False]*len(shortlist_rects)
    n_new_shortlist = []
    for k, rect_y_d in enumerate(zip(shortlist_rects, isDone)):
        rect_y, d = rect_y_d
        rect, y = rect_y
        if d == False:
            isDone[k] = True
            n_new_shortlist.append(rect_y)
            i = len(n_new_shortlist) - 1
            p1 = np.array(rect[0]+rect[2]/2,rect[1]+rect[3]/2)
            for n_k, n_r_y_d in enumerate(zip(shortlist_rects, isDone)):
                n_r_y, n_d = n_r_y_d
                r, n_y = n_r_y
                if n_d == False:
                    p2 = np.array(r[0]+r[2]/2,r[1]+r[3]/2)
                    if np.linalg.norm(p1-p2) < 2:
                        isDone[n_k] = True
                        #if np.max(y) < np.max(n_y):
                        if y < n_y:
                            n_new_shortlist[i] = n_r_y

    
    print("The new number of ROIs are:", len(n_new_shortlist))
    n_new_shortlist.sort(key=lambda x: get_y(x))
    
    print("Taking top 10")
    new_shortlist = []
    for abcd, y in n_new_shortlist[:10]:
        new_shortlist.append(abcd)
    print("The new number of ROIs are:", len(new_shortlist))
                        
    while True:
        # create a copy of original image
        imOut = im.copy()
 
        # itereate over all the region proposals
        for i, rect in enumerate(new_shortlist):
            #rect, y = rect_y
            # draw rectangle for region proposal till numShowRects
            if (i < numShowRects):
                x, y, w, h = rect
                cv2.rectangle(imOut, (x, y), (x+w, y+h), (0, 255, 0), 1, cv2.LINE_AA)
            else:
                break
 
        # show output
        imNewOut = cv2.resize(imOut, (newWidth*3, newHeight*3))
        cv2.imshow("Output", imOut)
 
        # record key press
        k = cv2.waitKey(0) & 0xFF
 
        # m is pressed
        if k == 109:
            # increase total number of rectangles to show by increment
            numShowRects += increment
        # l is pressed
        elif k == 108 and numShowRects > increment:
            # decrease total number of rectangles to show by increment
            numShowRects -= increment
        # q is pressed
        elif k == 113:
            break
    # close image show window
    #cv2.imwrite("selective.png",imOut)
    cv2.destroyAllWindows()
