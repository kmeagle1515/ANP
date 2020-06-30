import cv2
import numpy as np
import pytesseract
from pytesseract import Output
import glob
images= glob.glob(r"test\*.jpg")

for image in images:
  print(image)
  img=cv2.imread(image,0)
  ret,thresh = cv2.threshold(img,0,255,cv2.THRESH_OTSU)
  ##cv2.imshow("thresh",(thresh))
  ##cv2.waitKey(-1)

  kernel = np.ones((5,5),np.float32)/25
  dst = cv2.filter2D(thresh,-1,kernel)
  #cv2.imshow("3",dst)
  #cv2.waitKey(-1)

  contours,_=cv2.findContours(dst,cv2.RETR_TREE,cv2.CHAIN_APPROX_SIMPLE)
  cv2.drawContours(dst,contours,-1,(0,0,0),2)
  #cv2.imshow("3",dst)
  #cv2.waitKey(-1)
  kernel = np.array([[-1,-1,-1], [-1,9,-1], [-1,-1,-1]])
  im = cv2.filter2D(dst, -1, kernel)
  #cv2.imshow("3",im)
  #cv2.waitKey(-1)

  pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
  tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
  d = pytesseract.image_to_data(im,output_type=Output.DICT,lang="eng",config=tessdata_dir_config)
  
  for i in range(0, len(d["conf"])):
    if int(d["conf"][i]) >= 0:
      print(str(d["text"][i]).upper(),end=" ")
  print("")
  