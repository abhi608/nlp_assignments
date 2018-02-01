from __future__ import division
import sys
import numpy as np
from sklearn import svm
from sklearn.externals import joblib
from string import ascii_lowercase, ascii_uppercase

print sys.argv
file = open(sys.argv[1], 'r')

classifier = joblib.load('model.pkl') 

vocab = [] # for storing the vocab
arr = [] # for reading and storing characters of the text file
new_arr = [] # for storing the new text
# pre_process = [] # for storing the the vectorized training dataset
one_hot = { # for storing the indices of every character
}

# Filling the one_hot dict
count = 0
for c in ascii_lowercase:
	one_hot[c] = count
	count = count + 1

for c in ascii_uppercase:
	one_hot[c] = count
	count = count + 1

for c in range(10):
	one_hot[str(c)] = count
	count = count + 1

# print one_hot, len(one_hot)

# Read the text file character by character
while 1:
    char = file.read(1)
    if not char: break
    # print char
    arr.append(char)
    if char not in vocab:
    	vocab.append(char)
file.close()

# print("vocab size: ", len(vocab))

# add special characters to one_hot dict
tmp = ["\n","["," ",".","]","<","/",">",",",";",":","-","?","'","(",")","!","*"]

for c in tmp:
	one_hot[c] = count
	count = count + 1

# fill the new_arr
for i,_ in enumerate(arr):
	new_arr.append(arr[i]) # insert the charcters present in original text file
	if i < len(arr) - 4:
		if ((arr[i] == '.' or arr[i] == '!' or arr[i] == '?') and arr[i+1] != "'") or (arr[i] == "'" and (arr[i-1] == '.' or arr[i-1] == '!' or arr[i-1] == '?')):
			# select the window(4-3) characters
			char1 = arr[i+1]
			char2 = arr[i+2]
			char3 = arr[i+3]
			char_1 = arr[i]
			char_2 = arr[i-1]
			char_3 = arr[i-2]
			char_4 = arr[i-3]
			pre_process = [] # give this array as input to model
			obj = []
			tmp = [0]*len(one_hot)
			tmp[one_hot[char_4]] = 1
			obj.append(tmp)

			tmp = [0]*len(one_hot)
			tmp[one_hot[char_3]] = 1
			obj.append(tmp)

			tmp = [0]*len(one_hot)
			tmp[one_hot[char_2]] = 1
			obj.append(tmp)

			tmp = [0]*len(one_hot)
			tmp[one_hot[char_1]] = 1
			obj.append(tmp)

			tmp = [0]*len(one_hot)
			tmp[one_hot[char1]] = 1
			obj.append(tmp)

			tmp = [0]*len(one_hot)
			tmp[one_hot[char2]] = 1
			obj.append(tmp)

			tmp = [0]*len(one_hot)
			tmp[one_hot[char3]] = 1
			obj.append(tmp)

			pre_process.append(obj)

			pre_process = np.asarray(pre_process)
			# print pre_process.shape
			pre_process = pre_process.reshape(1, -1)
			# print pre_process.shape
			predicted = classifier.predict(pre_process) # Predict whether the current token is a sentence terminator or not

			# print predicted[0]
			# If current token is a sentence terminator, append "</s><s>" in the array
			if predicted[0] == 1:
				new_arr.append('</s><s>')
# Append "</s><s>" at the end of the file
new_arr.append('</s><s>')
file = open(sys.argv[2], "w")
for item in new_arr:
	file.write("%s" % item)
file.close()
		
