from PIL import Image
import cv2
import numpy as np
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
import os
import glob

def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

def canny(image,lower,upper):
    return cv2.Canny(image, lower, upper)


images= glob.glob(r"F:\AirPix\OCR\send\*.JPG")
print(len(images))

for jpg_image in images:
  image = cv2.imread(jpg_image)
  #cv2.imshow("original",image)
  #cv2.waitKey(-1)

  gray = get_grayscale(image)
  #cv2.imshow("gray",gray)
  #cv2.waitKey(-1)

  ret,thresh = cv2.threshold(gray,0,255,cv2.THRESH_OTSU)

  canny_1 = canny(gray,0.5*ret,ret)
  #cv2.imshow("edge",canny)
  #cv2.waitKey(-1)


  #find all your connected components (white blobs in your image)
  nb_components, output, stats, centroids = cv2.connectedComponentsWithStats(canny_1, connectivity=8)
  #connectedComponentswithStats yields every seperated component with information on each of them, such as size
  #the following part is just taking out the background which is also considered a component, but most of the time we don't want that.
  sizes = stats[1:, -1]; nb_components = nb_components - 1

  # minimum size of particles we want to keep (number of pixels)

  min_size = 50

  #your answer image
  img2 = np.zeros((output.shape))
  #for every component in the image, you keep it only if it's above min_size
  for i in range(0, nb_components):
      if min_size<= sizes[i]: #sizes[i] >= min_size
          img2[output == i + 1] = 255

  #cv2.imshow("final",img2)
  #cv2.waitKey(-1)

  directory= r"F:/AirPix/OCR/send_processed"
 
  file_name= jpg_image+"processed"+".jpg"
  cv2.imwrite(directory+file_name,img2)
  

#cv2.destroyAllWindows()


