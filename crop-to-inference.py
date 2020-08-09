import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import math

# Calculate distance between 2 points
def distance(pt1,pt2):
    return math.sqrt( (pt2[0]-pt1[0])*(pt2[0]-pt1[0]) + (pt2[1]-pt1[1])*(pt2[1]-pt1[1]) )

# function to preprocess image for Perspective correction
def Preprocess_Image(image):
    # convert image to gray
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY) 

    # convert image to binary
    ret, thresh = cv2.threshold(gray, 250, 255, cv2.THRESH_OTSU) 

    # get contour from binary
    contours, hierarchy = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE)

    # sort contours to get contour with max area
    cnt = sorted(contours, key=cv2.contourArea, reverse=True)[0] 

    # This section tries to find approx polygon that can fit the countour with
    # min curve seg == epsilon
    epsilon = 0.02 * cv2.arcLength(cnt, True) # 2% of the contour arc length
    approx_corners = cv2.approxPolyDP(cnt, epsilon, True)

    # to convert to list 
    pts = np.concatenate(approx_corners).tolist() 

    # width and height of cropped image
    w = image.shape[1]
    h = image.shape[0]

    #starting edge points as centre
    top_left_pt = (w/2,h/2)
    top_right_pt = (w/2,h/2)
    bottom_right_pt = (w/2,h/2)
    botton_left_pt = (w/2,h/2)

    #assign highest dimension width or height ( Mostly width)
    d_tl = w
    d_tr = w
    d_bl = w
    d_br = w

    for pt in pts:
        temp_d_tl = distance(pt,(0,0))
        temp_d_tr = distance(pt,(w,0))
        temp_d_bl = distance(pt,(0,h))
        temp_d_br = distance(pt,(w,h))
        
        # compare the points against edge points to get edge points
        if( temp_d_tl < d_tl ):
            d_tl = temp_d_tl
            top_left_pt = pt

        if( temp_d_tr < d_tr ):
            d_tr = temp_d_tr
            top_right_pt = pt
        
        if( temp_d_bl < d_bl ):
            d_bl = temp_d_bl
            botton_left_pt = pt
        
        if( temp_d_br < d_br ):
            d_br = temp_d_br
            bottom_right_pt = pt

    Edge_points = [top_left_pt,top_right_pt,bottom_right_pt,botton_left_pt]

    # get Width and height of the destination Box
    w1 = np.sqrt((Edge_points[0][0] - Edge_points[1][0]) ** 2 + (Edge_points[0][1] - Edge_points[1][1]) ** 2)
    w2 = np.sqrt((Edge_points[2][0] - Edge_points[3][0]) ** 2 + (Edge_points[2][1] - Edge_points[3][1]) ** 2)
    d_w = max(int(w1), int(w2))

    h1 = np.sqrt((Edge_points[0][0] - Edge_points[3][0]) ** 2 + (Edge_points[0][1] - Edge_points[3][1]) ** 2)
    h2 = np.sqrt((Edge_points[1][0] - Edge_points[2][0]) ** 2 + (Edge_points[1][1] - Edge_points[2][1]) ** 2)
    d_h = max(int(h1), int(h2))

    destination_corners = np.float32([(0, 0), (d_w - 1, 0), (d_w - 1, d_h - 1), (0, d_h - 1)])#no flipping and rotation require

    # Correction Matrix
    H, _ = cv2.findHomography(np.float32(Edge_points), destination_corners, method=cv2.RANSAC, ransacReprojThreshold=3.0)
    # Correct the Image
    image_unwarped = cv2.warpPerspective(image, H, (d_w, d_h), flags=cv2.INTER_LINEAR)

    print("Image is unwarped")

    return image_unwarped

# MAIN PROGRAM BEGINS HERE

# this is the image from Object Detector

image = cv2.imread('p550.jpg')

img  = Preprocess_Image(image)

cv2.imshow("Unwarped",img)
cv2.waitKey()