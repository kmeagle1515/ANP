
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

def predict_from_model(image,model,labels):
    image = cv2.resize(image,(80,80))
    image = np.stack((image,)*3, axis=-1)
    prediction = labels.inverse_transform([np.argmax(model.predict(image[np.newaxis,:]))])
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

labels = LabelEncoder()
labels.classes_ = np.load('license_character_classes.npy')
print("[INFO] Labels loaded successfully...")



image_paths = glob.glob("test/*.jpeg")

for i in range(85):
    try:
        test_image_path = image_paths[i]
        print('on image '+str(i))
        vehicle, LpImg,cor = get_plate(test_image_path)

        if (len(LpImg)): #check if there is at least one license image
            plate_image = cv2.convertScaleAbs(LpImg[0], alpha=(255.0))
        
            gray = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)
            blur = gray#cv2.GaussianBlur(gray,(7,7),0)
        
            binary = cv2.threshold(blur, 180, 255,
                                cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
            
            kernel3 = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
            thre_mor = cv2.morphologyEx(binary, cv2.MORPH_DILATE, kernel3)

        
        _, cont, _  = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        test_roi = plate_image.copy()

        crop_characters = []

        # define standard width and height of character plt.figure(figsize=(14,4))
    #grid = gridspec.GridSpec(ncols=len(crop_characters),nrows=1,figure=fig)

    #for i in range(len(crop_characters)):
    #    fig.add_subplot(grid[i])
    #   plt.axis(False)
    #   plt.imshow(crop_characters[i],cmap="gray")
    #plt.savefig("segmented_leter.png",dpi=300)    


    # ## Load pre-trained MobileNets model and predict


    # pre-processing input images and pedict with model


    #fig = plt.figure(figsize=(15,3))
    #cols = len(crop_characters)
    #grid = gridspec.GridSpec(ncols=cols,nrows=1,figure=fig)


        digit_w, digit_h = 30, 60

        for c in sort_contours(cont):
            (x, y, w, h) = cv2.boundingRect(c)
            ratio = h/w
            if 1<=ratio<=3.5: # Only select contour with defined ratio
                if h/plate_image.shape[0]>=0.5: # Select contour which has the height larger than 50% of the plate
                    # Draw bounding box arroung digit number
                    cv2.rectangle(test_roi, (x, y), (x + w, y + h), (0, 255,0), 2)

                    # Sperate number and gibe prediction
                    curr_num = thre_mor[y:y+h,x:x+w]
                    curr_num = cv2.resize(curr_num, dsize=(digit_w, digit_h))
                    _, curr_num = cv2.threshold(curr_num, 220, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
                    crop_characters.append(curr_num)

        final_string = ''
        for i,character in enumerate(crop_characters):
        #fig.add_subplot(grid[i])
            title = np.array2string(predict_from_model(character,model,labels))
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
