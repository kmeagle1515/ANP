
# remove warning message
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

# required library
import cv2
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from utils import detect_lp
from os.path import splitext,basename
from tensorflow.keras.models import model_from_json
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras.applications.mobilenet_v2 import preprocess_input
from sklearn.preprocessing import LabelEncoder
import glob
from skimage.filters import threshold_local
from skimage import measure
import imutils

from matplotlib import pyplot

def load_model(path):
    try:
        path = splitext(path)[0]
        with open('%s.json' % path, 'r') as json_file:
            model_json = json_file.read()
        model = model_from_json(model_json, custom_objects={})
        model.load_weights('%s.h5' % path)
        print("Loading model successfully...")
        return model
    except Exception as e:
        print(e)

def preprocess_image(image_path,resize=False):
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img / 255
    if resize:
        img = cv2.resize(img, (224,224))
    return img

def get_plate(image_path):
    Dmax = 608
    Dmin = 288
    vehicle = preprocess_image(image_path)
    ratio = float(max(vehicle.shape[:2])) / min(vehicle.shape[:2])
    side = int(ratio * Dmin)
    bound_dim = min(side, Dmax)
    _ , LpImg, _, cor = detect_lp(wpod_net, vehicle, bound_dim, lp_threshold=0.5)
    return vehicle, LpImg, cor

def sort_contours(cnts,reverse = False):
    i = 0
    boundingBoxes = [cv2.boundingRect(c) for c in cnts]
    (cnts, boundingBoxes) = zip(*sorted(zip(cnts, boundingBoxes),
                                        key=lambda b: b[1][i], reverse=reverse))
    return cnts

def predict_from_model(image,model,lb):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = lb.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
    return prediction



wpod_net_path = "wpod-net.json"
wpod_net = load_model(wpod_net_path)

# Load model architecture, weight and labels
json_file = open('MobileNets_character_recognition.json', 'r')
loaded_model_json = json_file.read()
json_file.close()
model = model_from_json(loaded_model_json)
model.load_weights("License_character_recognition_weight.h5")
print("[INFO] Model loaded successfully...")

lb = LabelEncoder()
lb.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")



image_paths = glob.glob("test/*.jpeg")

for i in range(85):
    try:
        test_image_path = image_paths[i]
        print('on image '+str(i))
        vehicle, LpImg,cor = get_plate(test_image_path)

        #image = cv2.imread('./op/23.png')
        #plt.imshow(image)
        #plt.show()
        image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        #image = cv2.cvtColor(LpImg[0],cv2.COLOR_RGB2BGR)
        V = cv2.split(cv2.cvtColor(image, cv2.COLOR_RGB2HSV))[2]
        T = threshold_local(V, 29, offset=15, method="gaussian")
        thresh = (V > T).astype("uint8") * 255
        thresh = cv2.bitwise_not(thresh)

        # resize the license plate region to a canonical size
        plate = imutils.resize(image, width=400)
        thresh = imutils.resize(thresh, width=400)
        #cv2.imshow("Thresh", thresh)
        #plt.imshow(thresh,cmap='gray')
        #plt.show()

        labels = measure.label(thresh, connectivity=2, background=0)
        charCandidates = np.zeros(thresh.shape, dtype="uint8")
        crop_characters = []
        for label in np.unique(labels):
            # if this is the background label, ignore it
            if label == 0:
                continue
            # otherwise, construct the label mask to display only connected components for the
            # current label, then find contours in the label mask
            labelMask = np.zeros(thresh.shape, dtype="uint8")
            labelMask[labels == label] = 255
            cnts = cv2.findContours(labelMask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
            cnts = cnts[0] if imutils.is_cv2() else cnts[1]
            #cnts = sort_contours(cnts)
            if len(cnts) > 0:
                # grab the largest contour which corresponds to the component in the mask, then
                # grab the bounding box for the contour
                c = max(cnts, key=cv2.contourArea)
                (boxX, boxY, boxW, boxH) = cv2.boundingRect(c)
                # compute the aspect ratio, solidity, and height ratio for the component
                aspectRatio = boxW / float(boxH)
                solidity = cv2.contourArea(c) / float(boxW * boxH)
                heightRatio = boxH / float(plate.shape[0])

                # determine if the aspect ratio, solidity, and height of the contour pass
                # the rules tests
                keepAspectRatio = aspectRatio < 1.0
                keepSolidity = solidity > 0.15
                keepHeight = heightRatio > 0.4 and heightRatio < 0.95

                # check to see if the component passes all the tests
                if keepAspectRatio and keepSolidity and keepHeight:
                    # compute the convex hull of the contour and draw it on the character
                    # candidates mask
                    #cv2.rectangle(thresh, (boxX, boxY), (boxX + boxW, boxY + boxH), (0, 255,0), 2)
                    curr_num = thresh[boxY:boxY+boxH,boxX:boxX+boxW]
                    curr_num = cv2.resize(curr_num, dsize=(30, 60))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append([curr_num,boxX])

        #charCandidates = segmentation.clear_border(charCandidates)
        #plt.imshow(thresh)
        #plt.imshow(image)
        crop_characters.sort(key = lambda x: x[1])
        final_string = ''
        for i,character in enumerate(crop_characters):
        #fig.add_subplot(grid[i])
            title = np.array2string(predict_from_model(character[0],model,lb))
            #plt.title('{}'.format(title.strip("'[]"),fontsize=20))
            final_string+=title.strip("'[]")
        #plt.axis(False)
        #plt.imshow(character,cmap='gray')

        #print(final_string)
        #plt.savefig('final_result.png', dpi=300)
        if final_string == '':
            final_string = str(i+1)
        pyplot.imsave(final_string+'.png',LpImg[0])
    except Exception as e:
        print('ERROR: '+ str(e))


# # The end!
