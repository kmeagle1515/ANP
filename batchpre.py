from PIL import Image
import cv2
import numpy as np
import pytesseract
import numpy as np
from PIL import ImageGrab
import time
import os
import glob
start_time = time.time()

# get grayscale image
def get_grayscale(image):
    return cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

#canny edge detection
def canny(image):
    return cv2.Canny(image, 100, 200)

#skew correction
def deskew(image):
    coords = np.column_stack(np.where(image > 0))
    angle = cv2.minAreaRect(coords)[-1]
    if angle < -45:
        angle = -(90 + angle)
    else:
        angle = -angle
    (h, w) = image.shape[:2]
    center = (w // 2, h // 2)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
    return rotated


images= glob.glob(r"F:\AirPix\OCR\dataset\*.PNG")
print(len(images))

for png_image in images: 
  image = cv2.imread(png_image)
  #cv2.imshow("original",image)
  #cv2.waitKey(-1)

  gray = get_grayscale(image)
  #cv2.imshow("gray",gray)
  #cv2.waitKey(-1)

  deskew_1 = deskew(gray)
  #cv2.imshow("deskew",deskew)
  #cv2.waitKey(-1)

  canny_1 = canny(deskew_1)
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

  directory=r"F:\AirPix\OCR\batch_processed"
  os.chdir(directory)
  file_name= png_image+".png"
  cv2.imwrite(file_name,img2)
  

#cv2.destroyAllWindows()

print("--- %s seconds ---" % (time.time() - start_time))