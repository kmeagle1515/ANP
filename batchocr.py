import cv2
import pytesseract
import pytesseract
import numpy as np
from PIL import ImageGrab
from pytesseract import Output
import time
import glob
start_time = time.time()

pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files (x86)\Tesseract-OCR\tesseract.exe'
tessdata_dir_config = r'--tessdata-dir "C:\Program Files (x86)\Tesseract-OCR\tessdata"'
#tessdata_dir_config = r'--tessdata-dir "<replace_with_your_tessdata_dir_path>"'
#img = cv2.imread(r'F:\AirPix\OCR\1.png')

images= glob.glob(r"F:\AirPix\OCR\batch_processed\*.PNG")
print(len(images))
fh= open(r"F:\AirPix\OCR\results.txt","w")
for image in images: 
  img = cv2.imread(image)
  img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
  d = pytesseract.image_to_data(img,output_type=Output.DICT,lang="eng",config=tessdata_dir_config)
  for i in range(0, len(d["conf"])):
    if int(d["conf"][i]) >= 45 :
      fh.write(str(image))
      fh.write("===>")
      fh.write(str(d["text"][i]).upper()+" ")
    
  fh.write("\n")