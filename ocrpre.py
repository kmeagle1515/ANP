from PIL import Image
import cv2
import numpy as np
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
import os
from pytesseract import Output
start_time = time.time()


#Set dpi of image 
def set_dpi(image):
  image.save(r"test-600.png", dpi=(300,300))

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# noise removal
def remove_noise(image):
    return cv2.medianBlur(image,5)
 
#thresholding
def thresholding(image):
    return cv2.adaptiveThreshold(image,255,cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY,11,2)

#dilation
def dilate(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.dilate(image, kernel, iterations = 1)
    
#erosion
def erode(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.erode(image, kernel, iterations = 1)

#opening - erosion followed by dilation
def opening(image):
    kernel = np.ones((5,5),np.uint8)
    return cv2.morphologyEx(image, cv2.MORPH_OPEN, kernel)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    gray = cv2.bitwise_not(gray)
    thresh = cv2.threshold(gray, 0, 255,
        cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
    coords = np.column_stack(np.where(thresh > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h),
        flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)    
    return rotated



image = cv2.imread(r"F:\AirPix\OCR\1.png")
cv2.imshow("1",image)
cv2.waitKey(-1)

gray = get_grayscale(image)
cv2.imshow("2",gray)
cv2.waitKey(-1)

canny = canny(gray)
cv2.imshow("2",canny)
cv2.waitKey(-1)



#find all your connected components (white blobs in your image)
nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny, connectivity=8)
#connectedComponentswithStats yields every seperated component with information on each of them, such as size
#the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
sizes = stats[1:, -1]; nb_components = nb_components - 1

# minimum size of particles we want to keep (number of pixels)

min_size = 50
max_size = 150
#your answer image
img2 = np.zeros((output.shape))
#for every component in the image, you keep it only if it's above min_size
for i in range(0, nb_components):
    if min_size<= sizes[i] <= max_size: #sizes[i] >= min_size
        img2[output == i + 1] = 255

cv2.imshow("final",img2)
cv2.waitKey(-1)

directory=r"F:\AirPix\OCR"
os.chdir(directory)
cv2.imwrite("processed.png",img2)

cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))

