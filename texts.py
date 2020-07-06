# import the necessary packages
from imutils.object_detection import non_max_suppression
import numpy as np
import time
import cv2

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from collections import namedtuple
from skimage.filters import threshold_local
from skimage import segmentation
from skimage import measure
from imutils import perspective
import numpy as np 
import imutils
import cv2
from PIL import Image
from keras.models import model_from_json
from sklearn.preprocessing import LabelEncoder
from sklearn.svm import SVC


image_read = r"lp2.jpg"
east = r"frozen_east_text_detection.pb"
min_confidence = 0.5
# load the input image and grab the image dimensions
image = cv2.imread(image_read)
orig = image.copy()
img = image.copy()
(H, W) = image.shape[:2]

# set the new width and height and then determine the ratio in change
# for both the width and height
(newW, newH) = (320, 320)
rW = W / float(newW)
rH = H / float(newH)

# resize the image and grab the new image dimensions
image = cv2.resize(image, (newW, newH))
(H, W) = image.shape[:2]

# define the two output layer names for the EAST detector model that
# we are interested -- the first is the output probabilities and the
# second can be used to derive the bounding box coordinates of text
layerNames = [
	"feature_fusion/Conv_7/Sigmoid",
	"feature_fusion/concat_3"]

# load the pre-trained EAST text detector
print("[INFO] loading EAST text detector...")
net = cv2.dnn.readNet(east)
                 
#net.setPreferableBackend(cv2.dnn.DNN_BACKEND_OPENCV)
net.setPreferableTarget(cv2.dnn.DNN_TARGET_CPU)

# construct a blob from the image and then perform a forward pass of
# the model to obtain the two output layer sets
blob = cv2.dnn.blobFromImage(image, 1.0, (W, H),
	(123.68, 116.78, 103.94), swapRB=True, crop=False)
start = time.time()
net.setInput(blob)
(scores, geometry) = net.forward(layerNames)
end = time.time()

# show timing information on text prediction
print("[INFO] text detection took {:.6f} seconds".format(end - start))

# grab the number of rows and columns from the scores volume, then
# initialize our set of bounding box rectangles and corresponding
# confidence scores
(numRows, numCols) = scores.shape[2:4] #80 x 80
rects = []
confidences = []

# loop over the number of rows
for y in range(0, numRows):
	# extract the scores (probabilities), followed by the geometrical
	# data used to derive potential bounding box coordinates that
	# surround text
	scoresData = scores[0, 0, y]
	xData0 = geometry[0, 0, y]
	xData1 = geometry[0, 1, y]
	xData2 = geometry[0, 2, y]
	xData3 = geometry[0, 3, y]
	anglesData = geometry[0, 4, y]

	# loop over the number of columns
	for x in range(0, numCols):
		# if our score does not have sufficient probability, ignore it
		if scoresData[x] < min_confidence:
			continue

		# compute the offset factor as our resulting feature maps will
		# be 4x smaller than the input image
		(offsetX, offsetY) = (x * 4.0, y * 4.0)

		# extract the rotation angle for the prediction and then
		# compute the sin and cosine
		angle = anglesData[x]
		cos = np.cos(angle)
		sin = np.sin(angle)

		# use the geometry volume to derive the width and height of
		# the bounding box
		h = xData0[x] + xData2[x]
		w = xData1[x] + xData3[x]

		# compute both the starting and ending (x, y)-coordinates for
		# the text prediction bounding box
		endX = int(offsetX + (cos * xData1[x]) + (sin * xData2[x]))
		endY = int(offsetY - (sin * xData1[x]) + (cos * xData2[x]))
		startX = int(endX - w)
		startY = int(endY - h)

		# add the bounding box coordinates and probability score to
		# our respective lists
		rects.append((startX, startY, endX, endY))
		confidences.append(scoresData[x])

# apply non-maxima suppression to suppress weak, overlapping bounding
# boxes
boxes = non_max_suppression(np.array(rects), probs=confidences)

# loop over the bounding boxes
text = list()
for (startX, startY, endX, endY) in boxes:
  # scale the bounding box coordinates based on the respective
  # ratios
  startX = int(startX * rW)
  startY = int(startY * rH)
  endX = int(endX * rW)
  endY = int(endY * rH)

  # draw the bounding box on the image

  text.append((startX, startY, endX, endY))

text_copy = list()

for text_box in text:
  length = abs(text_box[0]- text_box[2])
  breadth = abs(text_box[1]- text_box[3])
  area = length * breadth
  if area > 2500:
    text_copy.append(text_box)

for box in text_copy:
  cv2.rectangle(orig, (box[0], box[1]), (box[2], box[3]), (0, 255, 0), 2)


# show the output image
#cv2.imshow("Text Detection", orig)
#cv2.waitKey(-1)

text_copy.sort(key = lambda text:text[1])


def detectCharacterCandidates(text_line):
  # extract the Value component from the HSV color space and apply adaptive thresholding
  # to reveal the characters on the license plate
  V = cv2.split(cv2.cvtColor(text_line, cv2.COLOR_BGR2HSV))[2]
  blur = cv2.GaussianBlur(V,(3,3),0)
  th3 = cv2.adaptiveThreshold(blur,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C,cv2.THRESH_BINARY,31,5)
  thresh = cv2.bitwise_not(th3) 
  #T = threshold_local(blur, 29, offset=15, method="gaussian")
  #thresh = (V > T).astype("uint8") * 255
  #thresh = cv2.bitwise_not(thresh)

  # resize the license plate region to a canonical size
  plate = imutils.resize(text_line, width=400)
  thresh = imutils.resize(thresh, width=400)

  # perform a connected components analysis and initialize the mask to store the locations
  # of the character candidates
  labels = measure.label(thresh, connectivity=2, background=0)
  charCandidates = np.zeros(thresh.shape, dtype="uint8")
  # loop over the unique components
  ROI_number = 0
  ROI_list = list()
  for label in np.unique(labels):
    # if this is the background label, ignore it
    if label == 0:
      continue

    # otherwise, construct the label mask to display only connected components for the
    # current label, then find contours in the label mask
    labelMask = np.zeros(thresh.shape, dtype="uint8")
    labelMask[labels == label] = 255
    cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cnts = cnts[0] #if imutils.is_cv2() else cnts[1]
    # ensure at least one contour was found in the mask
    if len(cnts) > 0:
      # grab the largest contour which corresponds to the component in the mask, then
      # grab the bounding box for the contour
      #c = max(cnts, key=cv2.contourArea)
      c=max(cnts)
      (x, y, w, h) = cv2.boundingRect(c)
      

      # compute the aspect ratio, solidity, and height ratio for the component
      aspectRatio = w / float(h)
      solidity = cv2.contourArea(c) / float(w* h)
      #Solidity (convexity) of an image object is, area of the image object divided by area of its convex hull
      heightRatio = h / float(plate.shape[0])

      # determine if the aspect ratio, solidity, and height of the contour pass
      # the rules tests
      keepAspectRatio = aspectRatio < 1.0
      keepSolidity = solidity > 0.15
      keepHeight = heightRatio > 0.4 and heightRatio < 0.95

      # check to see if the component passes all the tests
      if keepAspectRatio and keepSolidity and keepHeight:
        # compute the convex hull of the contour and draw it on the character
        # candidates mask
        ROI = thresh[y:y+h, x:x+w]
        ROI_list.append((x,ROI))
        #hull = cv2.convexHull(c)
        #cv2.drawContours(charCandidates, [hull], -1, 255, -1)     
  # clear pixels that touch the borders of the character candidates mask and detect
  # contours in the candidates mask
  charCandidates = segmentation.clear_border(charCandidates)

  # There will be times when we detect more than the desired number of characters --
  # it would be wise to apply a method to 'prune' the unwanted characters

  # return the license plate region object containing the license plate, the thresholded
  # license plate, and the character candidates
  #cv2.imshow("Segmentation",charCandidates)
  #cv2.waitKey(-1)
  ROI_list.sort(key = lambda ROI:ROI[0])
  return ROI_list

json_file = open(r'F:\AirPix\OCR\OCR end-to-end\MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights(r"License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

labels = LabelEncoder()
print(labels)
labels.classes_ = np.load(r'license_character_classes.npy')
print("[INFO] Labels loaded successfully...")

def predict_from_model(ROI_list,model,labels):
  for charac in ROI_list:
    charac = cv2.resize(charac[1],(80,80))
    charac = np.stack((charac,)*3, axis=-1)

    prediction = labels.inverse_transform([np.argmax(model.predict(charac[np.newaxis,:]))])
    title = np.array2string(prediction)
    print(title.strip("'[]"),end="")
    
    q = model.predict(charac[np.newaxis,:])

    probs = np.argpartition(q[0], -4)[-4:]
    
    
    prob_char = labels.inverse_transform(probs)
    title = np.array2string(prob_char)
    print(title)

    #prediction = labels.inverse_transform([np.argmax(model.predict(charac[np.newaxis,:]))])
    
    #title = np.array2string(prediction)
    #print(title.strip("'[]"),end="")

for text in text_copy:
  x1 = text[0]
  y1 = text[1]
  x2 = text[2]
  y2 = text[3]
  roi = img[y1:y2,x1:x2]
  ROI_list = detectCharacterCandidates(roi)
  predict_from_model(ROI_list,model,labels)
    
 

