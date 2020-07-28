import itertools
import re
import time
import numpy as np
import math
start = time.time() #timing program

#list of top 3 probablities after OCR
all_probs = [{'M': 0.7573969, 'D': 0.09272581, 'N': 0.051710963}, 
            {'H': 1.0, 'L': 1.2237347e-09, 'V': 3.8113038e-10}, 
            {'0': 0.99945205, 'D': 0.00032163598, 'O': 0.00015011593}, 
            {'1': 0.9999995, 'I': 1.8725837e-07, 'J': 1.8035753e-07}, 
            {'A': 0.99999964, 'H': 2.596042e-07, 'K': 6.765525e-08}, 
            {'B': 0.99999976, '8': 2.6263254e-07, 'E': 2.8750369e-09}, 
            {'1': 1.0, '7': 6.912726e-09, 'I': 6.5992776e-09},
            {'2': 0.9999882, 'Z': 1.0806609e-05, 'P': 7.298813e-07}, 
            {'3': 1.0, 'Z': 4.48661e-10, 'D': 1.5507319e-10}, 
            {'A': 0.9968413, '4': 0.0016599022, 'H': 0.0003524768}]

#Regex Template 
template = "^[A-Z]{2}[0-9]{2}[A-Z]{2}[0-9]{4}$"

#List of state code
state_code = ['AP','AR','AS','BR','CG','GA','GJ','HR','HP','JH','KA','KL','MP',
              'MH','MN','ML','MZ','NL','OD','PB','RJ','SK','TN','TR','UP','UK',
              'UA','WB','TS','AN','CH','DN','DD','JK','LA','LD','DL','PY']

list_keys = list()
for x in all_probs:
  list_keys.append(list(x.keys())) #Making a list of keys 

all_perms = list(itertools.product(*list_keys)) #all possible permutations of a numberplate from the top three probabilities.

conf_final = {}
for each_perm in all_perms:
  str1 = ""
  confidence  = 0
  str1 = str1.join(each_perm) #converting ("M","H","0","1",...) to "MH01.."
  if re.match(template,str1) is not None and str1[0:2] in state_code: #Pattern matching with the given template
    for index, x in enumerate(each_perm):
      confidence += float(all_probs[index][x]) #Summing confidence for each character 
      conf_final[str1] = (confidence,1) # 1 --> Match 

conf_final = {k: v for k, v in sorted(conf_final.items(), key=lambda item: item[1][0],reverse= True)} #Sorting the numberplate based on confidence
#out = dict(list(conf_final.items())[0: 10])  
count = 0 
topN = 20

#Printing the top N matches 
s = sum(np.exp(np.ravel([[v[0] for k,v in conf_final.items()]]))) #sum of all valid confidences 
for key, value in conf_final.items():
  number_plate = key
  prob = value[0]
  pattern = value[1]
  if count< topN:
    if pattern == 1 :
      conf = (math.exp(prob))/s #soft max.. convert to percentage
      conf = conf*100
      print(number_plate+"   "+"Confidence ="+f"{str(round(conf,3)): <5}"+"%"+"    "+"Match = True")
      count += 1
  else:
    break


print(f"--- {time.time() - start} seconds ---") #Total time for program (0.05 - 0.08 secs)

#OUTPUT

'''
MH01AB1234   Confidence =17.708%    Match = True
MH01HB1234   Confidence =6.514%    Match = True
MH01KB1234   Confidence =6.514%    Match = True
MH01AE1234   Confidence =6.514%    Match = True
MH01AB7234   Confidence =6.514%    Match = True
ML01AB1234   Confidence =6.514%    Match = True
DL01AB1234   Confidence =3.351%    Match = True
NL01AB1234   Confidence =3.217%    Match = True
MH01HE1234   Confidence =2.397%    Match = True
MH01KE1234   Confidence =2.397%    Match = True
MH01HB7234   Confidence =2.397%    Match = True
ML01HB1234   Confidence =2.397%    Match = True
MH01KB7234   Confidence =2.397%    Match = True
ML01KB1234   Confidence =2.397%    Match = True
MH01AE7234   Confidence =2.397%    Match = True
ML01AE1234   Confidence =2.397%    Match = True
ML01AB7234   Confidence =2.397%    Match = True
DL01HB1234   Confidence =1.233%    Match = True
DL01KB1234   Confidence =1.233%    Match = True
DL01AE1234   Confidence =1.233%    Match = True
--- 0.08078408241271973 seconds ---

'''







