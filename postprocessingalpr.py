import itertools
import re
import time
start = time.time() #timing program

#list of top 3 probablities after OCR
all_probs = [{'M': 0.7573969, 'D': 0.09272581, 'N': 0.051710963, }, 
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


conf = {}
for each_perm in all_perms:
  confidence = 0
  for index, x in enumerate(each_perm):
    confidence += float(all_probs[index][x]) #assigning confidence to each permutation (NEED TO OPTIMISE)
  conf[each_perm] = confidence


conf_final = {} 
for key,value in conf.items():
  str1 = ""
  str1 = str1.join(key) #converting ("M","H","0","1",...) to "MH01.."
  if re.match(template,str1) is not None and str1[0:2] in state_code: #Pattern matching with the given template
    conf_final[str1] = (value,1) # 1 --> Match 
  else:
    conf_final[str1] = (value,0) #0 --> Not a Match

conf_final = {k: v for k, v in sorted(conf_final.items(), key=lambda item: item[1][0],reverse= True)} #Sorting the numberplate based on confidence
#out = dict(list(conf_final.items())[0: 10])  
count = 0 
topN = 20

#Printing the top N matches 
for key, value in conf_final.items():
  number_plate = key
  prob = value[0]
  pattern = value[1]
  if count< topN:
    if pattern == 1 :
      print(number_plate+"   "+"Confidence ="+f"{str(round(prob,3)): <5}"+"   "+"Match = True")
      count += 1
  else:
    break


print(f"--- {time.time() - start} seconds ---") #Tatal time for program




