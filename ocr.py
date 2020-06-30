import cv2
import pytesseract
import pytesseract
import numpy as np
from PIL import ImageGrab
from pytesseract import Output
import time
start_time = time.time()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
#tessdata_dir_config = r'--tessdata-dir "<replace_with_your_tessdata_dir_path>"'
#img = cv2.imread(r'F:\AirPix\OCR\1.png')
img = cv2.imread(r"F:\AirPix\OCR\processed.png")

img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

d = pytesseract.image_to_data(img,output_type=Output.DICT,lang="eng",config=tessdata_dir_config)

for i in range(0, len(d["conf"])):
  if int(d["conf"][i]) >= 30 :
    print(str(d["text"][i]).upper(),end=" ")

#cv2.imshow('img', img)
#cv2.waitKey(5000)

""" fix DPI (if needed) 300 DPI is minimum
fix text size (e.g. 12 pt should be ok)
try to fix text lines (deskew and dewarp text)
try to fix illumination of image (e.g. no dark part of image)
binarize and de-noise image """

print("--- %s seconds ---" % (time.time() - start_time))