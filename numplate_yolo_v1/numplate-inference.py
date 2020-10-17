import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image
import math
import imutils
from tensorflow.keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
import heapq
import itertools
import re
import scipy.fftpack

from tool.utils import *
from tool.torch_utils import *
from tool.darknet2pytorch import Darknet
import argparse

"""hyper parameters"""
use_cuda = True
m = None # darknet model

# Initialise Darknet with yolo
def init_darknet(cfgfile, weightfile):
    global m , use_cuda
    m = Darknet(cfgfile)

    m.print_network()
    m.load_weights(weightfile)
    print('Loading weights from %s... Done!' % (weightfile))

    if use_cuda:
        m.cuda()

# detect Plate regions and return the bounding boxes
def detect_numplate(img):
    global m , use_cuda

    sized = cv2.resize(img, (m.width, m.height))
    sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

    boxes = do_detect(m, sized, 0.4, 0.6, use_cuda)

    return boxes

# get cropped plate image array
def get_cropped_plates(img,boxes):
    width = img.shape[1]
    height = img.shape[0]

    plate_area = []
  
    for i in range(len(boxes[0])):        
        box = boxes[0][i]
        # print("box",box)
        x1 = int(box[0] * width)
        y1 = int(box[1] * height)
        x2 = int(box[2] * width)
        y2 = int(box[3] * height)
        crop_img = img[y1:y2, x1:x2]

        plate_area.append(crop_img)

    return plate_area

    

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
    # Given a binary image, first find all of its contours
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
    #Each contour detected as border is drawn over with black to remove it.
    for idx in contourList:
        cv2.drawContours(imgBWcopy, contours, idx, (0,0,0), -1)

    return imgBWcopy


# Detect Lines
def Multiline_detection(image):
    h = image.shape[0]
    w = image.shape[1]
    
    # resize image for smooth out the image
    image = cv2.resize(image, (400,int(400*h/w)))


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
        sum_of_pixels_in_row = np.sum(mask,axis = 1)
        
        # Max Value in the ROW
        m = np.max(sum_of_pixels_in_row)
        

        # In here we select one row at a time 
        # Normalise it to a value based on the max value in that row
        # and then stor the sum in p_sum 
        # and that sum will later be compared against all colors to choose
        p_sum = 0
        for row in range(sum_of_pixels_in_row.shape[0]):
            p = int(sum_of_pixels_in_row[row]*400/m)
            p_sum+=p
        # append p_sum to v
        sum_of_individual_colors.append(p_sum)
    
    # choose the color which returns max sum
    bg = color[sum_of_individual_colors.index(max(sum_of_individual_colors))]
    
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


    sum_over_row = []
    # for each row..
    for row in range(binary.shape[0]):
        #normalize sum of pixel values in the row
        p = int(proj[row]*w/m)
        if p<75:
            p=0
        elif p>150:
            p=150
        #append sum of each row to vector with sum value and row number
        sum_over_row.append([p,row])

    i=0
    #interate over list of sums until we get row number of horizontal top boundary, i.e we reach first row with 0 white pixels
    while sum_over_row[i][0]!=0:
        i+=1
    #crop image vertically until top horzontal boundary is removed
    sum_over_row = sum_over_row[i:len(sum_over_row)]

    transition_points = []
    #find all the transition points in order to determine boundaries of each individual line in numberplate
    for i in range(1,len(sum_over_row)):
        #find transition point from black to white, i.e top of each individual line in numberplate
        if sum_over_row[i][0]>0 and sum_over_row[i-1][0] == 0:
            transition_points.append(sum_over_row[i][1])
        #find transition point from white to bloack, i.e bottom of each individual line in numberplate
        elif sum_over_row[i][0]==0 and sum_over_row[i-1][0] > 0:
            transition_points.append(sum_over_row[i-1][1])

    end = len(transition_points)-1

    # List of Individual lines
    individual_lines = []
    for i in range(0,end,2):
        #ignore all white transition points/detected numberplate line which have a thickness less than 15
        #this is done to get rid of unnecessary noise detected as character line
        if transition_points[i+1]-transition_points[i] < 15:
            #i = i+2
            continue

        if(i+1 < len(transition_points)):
            #slicing binary image to get individual numberplate line
            temp=binary[transition_points[i]-10:transition_points[i+1]+10,:]
            #removing any additional border from detected individual line and appending to list
            individual_lines.append(imclearborder(temp,1))

    # Returns list of lines in the licence plate
    return individual_lines

# Character Segmentation on lines
def Segmentation(individual_lines):
    character_width=[]
    crop_characters = []
    
    for line in individual_lines:
        if line.any():
            # Create sort_contours() function to grab the contour of each digit from left to right
            # so as to maintain order of individual characters
            def sort_contours(cnts,reverse = False):
                i = 0
                boundingBoxes = [cv2.boundingRect(c) for c in cnts]
                (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                                    key=lambda b: b[1][i], reverse=reverse))
                return cnts


            _, cont, _  = cv2.findContours(line, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for c in sort_contours(cont):
                (x, y, w, h) = cv2.boundingRect(c)
                #Detected contour is character only if it's height is more than 55% of the individual line height
                if h/line.shape[0]>=0.55:
                    character_width.append(w)
                    #slice line into individual characters
                    curr_num = line[y:y+h,x:x+w]
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

    return crop_characters,character_width

# Splits segmented character image if multiple characters in single character image
def Splitting_Characters(crop_characters,character_width):
    crop_characters_final=[]
    #if any character is split then list of character width will also have to change
    updated_character_width=[]
    #variable is used to find running avg of width of segmented characters
    avg=character_width[0]
    for i in range(1,len(character_width)+1):
        width_ratio = (character_width[i-1]/avg)
        avg=sum(character_width[0:i])/len(character_width[0:i])
        
        #if character width is less than 1.5 of avg width then it is individual character
        if width_ratio<1.5:
            crop_characters_final.append(crop_characters[i-1])
            updated_character_width.append(character_width[i-1])
        #else we split the image into 2 separate characters at the global minima by using vertical projection
        else:
            #sum of values over column ie vertically
            counts = np.sum(crop_characters[i-1]!=0, axis=0)
            row_number = [i for i in range(crop_characters[i-1].shape[1])]
            #convolve the sum to get a smoother projection
            counts = np.convolve(counts, np.ones(1), mode='same')
            #find minimas in the projection  
            minm,_ = scipy.signal.find_peaks(-counts)
            #find the global minima, i.e the minima point with lowest value
            m=minm[0]
            for a in range(1,len(minm)):
                if counts[m] > counts[minm[a]]:
                    m=minm[a]
            #split the character image into 2 at the point of global minima
            c1 = crop_characters[i-1][:,:m]
            c2 = crop_characters[i-1][:,m:]
            crop_characters_final.append(c1)
            updated_character_width.append(m)
            crop_characters_final.append(c2)
            updated_character_width.append(character_width[i-1]-m)

    crop_characters=crop_characters_final
    character_width=updated_character_width
    #we perform the same process as above again with all characters in reverse. 
    #this is done to take care of cases when the first or last segmented character is the one having multiple characters
    crop_characters.reverse()
    character_width.reverse()
    crop_characters_final=[]
    avg=character_width[0]
    for i in range(1,len(character_width)+1):
        v = (character_width[i-1]/avg)
        avg=sum(character_width[0:i])/len(character_width[0:i])
        
        if v<=1.5:
            crop_characters_final.append(crop_characters[i-1])
        else:
            counts = np.sum(crop_characters[i-1]!=0, axis=0)
            row_number = [i for i in range(crop_characters[i-1].shape[1])]
            counts = np.convolve(counts, np.ones(1), mode='same')
            minm,_ = scipy.signal.find_peaks(-counts)
            m=minm[0]
            for a in range(1,len(minm)):
                if counts[m] > counts[minm[a]]:
                    m=minm[a]
            c1 = crop_characters[i-1][:,:m]
            c2 = crop_characters[i-1][:,m:]
            crop_characters_final.append(c2)
            crop_characters_final.append(c1)
    crop_characters=crop_characters_final
    crop_characters.reverse()
    character_width.reverse()
    return crop_characters

#loads model and weights for OCR and returns model and LabelEncoder()
def Load_ocr_model():
    # Load model architecture, weight and labels
    json_file = open('MobileNets_character_recognition.json', 'r')
    loaded_model_json = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model_json)
    model.load_weights("License_character_recognition_weight.h5")

    label_encoder = LabelEncoder()
    label_encoder.classes_ = np.load('license_character_classes.npy')
    return model,label_encoder

#Perform ocr and return predicted number plate string 
def Perform_ocr(crop_characters,model,label_encoder):
  max_prob_list = list()
  for i,character in enumerate(crop_characters):

    #resize cropped image to 80x80 (fixed input size to model)
    character = cv2.resize(character,(80,80)) 

    #joins array sequence about last dimension
    character = np.stack((character,)*3, axis=-1)

    #prediction is a vector, size 36 , of probabilities for each label (total 36 labels --> 10 digits (0-9) and 26 alphabets)
    prediction = model.predict(character[np.newaxis,:])
    
    #max_prob_index --> indices of top 3 values (probabilities) from prediction vector
    max_prob_index = heapq.nlargest(3,range(len(prediction[0])),key = prediction[0].__getitem__)
    
    #max_prob_char --> converting index to the corresponding label (character decoding)
    max_prob_char = label_encoder.inverse_transform(max_prob_index)

    #max_prob_dict --> a dictionary where key is the character and value is probability. 
    max_prob_dict = {max_prob_char[i]:prediction[0][max_prob_index[i]] for i in range(len(max_prob_char))}

    #max_prob_list --> appending the top 3 character:probility pair for each cropped character
    max_prob_list.append(max_prob_dict)

  return max_prob_list

#Perform Post Processing and return the top "N" numberplates using the given template
def Post_processing(max_prob_list):
  topN_list = list() #list containing top N number plates with confidences
  #List of Regex Template 
  template_list = ["^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$","^[A-Z]{2}[0-9]{2}[A-Z]{3}[0-9]{3}$","^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{3}$"]
  template = '(?:% s)' % '|'.join(template_list) 
  #List of state code
  state_code = ['AP','AR','AS','BR','CG','GA','GJ','HR','HP','JH','KA','KL','MP',
                'MH','MN','ML','MZ','NL','OD','PB','RJ','SK','TN','TR','UP','UK',
                'UA','WB','TS','AN','CH','DN','DD','JK','LA','LD','DL','PY']

  list_keys = list()

  for x in max_prob_list:
    #a list of keys to use with itertools module
    list_keys.append(list(x.keys())) 

  #all possible permutations of a numberplate from the top three probabilities.
  all_perms = list(itertools.product(*list_keys)) 

  valid_numberplates = {} #dictionary to store valid numberplates and their confidence after template matching
  for each_perm in all_perms:
    str1 = ""
    confidence  = 0

    #converting ("M","H","0","1",...) to "MH01.."
    str1 = str1.join(each_perm) 

    if re.match(template,str1) is not None and str1[0:2] in state_code: #Pattern matching with the given template
      for index, x in enumerate(each_perm):
        confidence += float(max_prob_list[index][x]) #Summing confidence for each character 
        valid_numberplates[str1] = (confidence,1) # 1 --> Match 

  #Sorting the numberplate based on confidence values
  valid_numberplates = {k: v for k, v in sorted(valid_numberplates.items(), key=lambda item: item[1][0],reverse= True)} 
    
  count = 0 
  topN = 20 #Setting the number of top matches 

  #Printing the top N matches 

  #sum of all valid confidences (used for softmax)
  sum_valid_confidences = sum(np.exp(np.ravel([[v[0] for k,v in valid_numberplates.items()]]))) 
  for key, value in valid_numberplates.items():
    number_plate = key
    probability = value[0]
    pattern = value[1]
    if count< topN:
      if pattern == 1 :
        accuracy = (math.exp(probability))/sum_valid_confidences #taking soft max and converting to percentage
        accuracy = accuracy*100 
        topN_list.append({number_plate:accuracy})
        
        #print(number_plate+"   "+"Confidence ="+f"{str(round(accuracy,3)): <5}"+"%"+"    "+"Match = True")
        
        count += 1
    else:
      break
  return topN_list

# MAIN PROGRAM BEGINS HERE

# this is the image from Object Detector

image = cv2.imread('numplate/3.jpg')

init_darknet('numplate/numplate.cfg', 'numplate/numplate.weights') 

boxes = detect_numplate(image)

cropped_plate = get_cropped_plates(image,boxes)

for plate in cropped_plate:
    img  = Preprocess_Image(plate)

    cv2.imshow("Unwarped",img)
    cv2.waitKey()