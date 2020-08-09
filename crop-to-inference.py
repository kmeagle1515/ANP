import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import imutils

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


# Removes the border from binary image
def imclearborder(imgBW, radius):
    # Given a black and white image, first find all of its contours
    imgBWcopy = imgBW.copy()
    _,contours,hierarchy = cv2.findContours(imgBWcopy.copy(), cv2.RETR_LIST, 
        cv2.CHAIN_APPROX_SIMPLE)

    # Get dimensions of image
    imgRows = imgBW.shape[0]
    imgCols = imgBW.shape[1]    

    contourList = [] # ID list of contours that touch the border

    # For each contour...
    for idx in np.arange(len(contours)):
        # Get the i'th contour
        cnt = contours[idx]

        # Look at each point in the contour
        for pt in cnt:
            rowCnt = pt[0][1]
            colCnt = pt[0][0]

            # If this is within the radius of the border
            # this contour goes
            check1 = (rowCnt >= 0 and rowCnt < radius) or (rowCnt >= imgRows-1-radius and rowCnt < imgRows)
            check2 = (colCnt >= 0 and colCnt < radius) or (colCnt >= imgCols-1-radius and colCnt < imgCols)

            if check1 or check2:
                contourList.append(idx)
                break

    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


# Detect Lines
def Multiline_detection(img):
    h = img.shape[0]
    w = img.shape[1]
    
    # resize image for smooth out the image
    img = cv2.resize(img, (400,int(400*h/w)))

    kernel = np.ones((5,5),np.uint8)
    # change contrast
    img = 0.5*img + 0.5*cv2.dilate(img,kernel)

    # bgcolor detection:
    # List of RGB color sections to check for plate color
    boundaries = [
        ([99,5,0], [255,127,120]), #red
        ([6, 66, 9],[117,255,136]), #green
        ([135, 90, 0], [255,255,61]),   #yellow
        ([0, 0, 0], [69, 69, 69]),  #black
        ([100, 100, 100], [255,255,255]),   #white
    ]

    color=['red','green','yellow','black','white']

    # loop over the boundaries    
    sum_of_individual_colors = []
    for (lower, upper) in boundaries:
        # create NumPy arrays from the boundaries
        lower = np.array(lower, dtype = "uint8")
        upper = np.array(upper, dtype = "uint8")

        # find the colors within the specified boundaries and apply
        # check all pixels thats fall within the RGB range
        mask = cv2.inRange(image, lower, upper)
        
        # Sum of all the colums
        proj = np.sum(mask,axis = 1)
        
        # Max Value in the ROW
        m = np.max(proj)
        

        # In here we select one row at a time 
        # Normalise it to a value based on the max value in that row
        # and then stor the sum in p_sum 
        # and that sum will later be compared against all colors to choose
        p_sum = 0
        for row in range(proj.shape[0]):
            p = int(proj[row]*400/m)
            p_sum+=p
        # append p_sum to v
        sum_of_individual_colors.append(p_sum)
    
    # choose the color which returns max sum
    bg = color[sum_of_individual_colors.index(max(v))]
    
    # Invert the image if the background is dark compared to characters
    if(bg == 'red' or bg == 'green'):
        image=-image
    
    # Binarisation
    # convert image to HSV and select only V ( Intensity image )
    V = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[2]
    # thresh to binary
    _,binary = cv2.threshold(V,0,255,cv2.THRESH_BINARY_INV+cv2.THRESH_OTSU)


    # MULTILINE DETECTION

    # Sum of Rows  ==> Column vector
    proj = np.sum(binary,axis = 1)

    m = np.max(proj)
    w = 500


    pts = []
    # Draw a line for each row
    for row in range(binary.shape[0]):
        p = int(proj[row]*w/m)
        if p<75:
            p=0
        elif p>150:
            p=150
        pts.append([p,row])

    i=0
    while pts[i][0]!=0:
        i+=1

    pts = pts[i:len(pts)]

    c = []
    for i in range(1,len(pts)):
        if pts[i][0]>0 and pts[i-1][0] == 0:
            c.append(pts[i][1])
        elif pts[i][0]==0 and pts[i-1][0] > 0:
            c.append(pts[i-1][1])

    end = len(c)-1

    # List of Individual lines
    bc = []
    for i in range(0,end,2):
        if c[i+1]-c[i] < 15:
            #i = i+2
            continue
        if(i+1 < len(c)):
            temp=binary[c[i]-10:c[i+1]+10,:]
            bc.append(imclearborder(temp,1))

    # Returns list of lines in the licence plate
    return bc

# Character Segmentation on lines
def Segmentation(bc):
    t1=[]
    crop_characters = []
    
    for b in bc:
        if b.any():
            bc2 = b
            # Create sort_contours() function to grab the contour of each digit from left to right
            def sort_contours(cnts,reverse = False):
                i = 0
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                    key=lambda b: b[1][i], reverse=reverse))
                return cnts


            _, cont, _  = cv2.findContours(bc2, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            plate_image = bc2
            # creat a copy version "test_roi" of plat_image to draw bounding box
            test_roi = plate_image.copy()

            # Initialize a list which will be used to append charater image


            # define standard width and height of character
            digit_w, digit_h = 30, 60
            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                ratio = h/w
                if True:# Only select contour with defined ratio
                   # print(h/plate_image.shape[0])
                    if h/plate_image.shape[0]>=0.55:
                        t1.append(w)
                        # Select contour which has the height larger than 50% of the plate
                        # Sperate number and gibe prediction
                        curr_num = bc2[y:y+h,x:x+w]
                        #curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                        _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                        crop_characters.append(curr_num)



# MAIN PROGRAM BEGINS HERE

# this is the image from Object Detector

image = cv2.imread('p550.jpg')

img  = Preprocess_Image(image)

cv2.imshow("Unwarped",img)
cv2.waitKey()