import json
from string import ascii_lowercase, ascii_uppercase

file = open('training_data.txt', 'r')

vocab = [] # for storing the vocab
arr = [] # for reading and storing characters of the text file
pre_process = [] # for storing the the vectorized training dataset
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
print one_hot, len(one_hot)

while 1:
    char = file.read(1)          # read by character
    if not char: break
    # print char
    arr.append(char)
    if char not in vocab:
    	vocab.append(char)
file.close()
print vocab
print("vocab size: ", len(vocab))

# add special characters to one_hot dict
tmp = ["\n","["," ",".","]","<","/",">",",",";",":","-","?","'","(",")","!","*"]

for c in tmp:
	one_hot[c] = count
	count = count + 1

# print arr

print len(one_hot)

# fill the pre_process array
for i,_ in enumerate(arr):
	# print c, i
	if i < len(arr)-5:
		# print "test", i, arr[i]
		# store the characters in left and right window(current window size is 4-3)
		if (arr[i] == '.' or arr[i] == '?' or arr[i] == '!') and arr[i+1] == "'" and arr[i+2] != '<' and arr[i+3] != '/' and arr[i+4] != 's':
			char1 = arr[i+2]
			char2 = arr[i+3]
			char3 = arr[i+4]
			char4 = arr[i+5]
			char_1 = arr[i+1]
			char_2 = arr[i]
			char_3 = arr[i-1]
			char_4 = arr[i-2]
			# print arr[i-3], arr[i-2], arr[i-1], arr[i], arr[i+1], arr[i+2], arr[i+3]
			# print "test1", arr[i], char1, char2, char3, char_1, char_2, char_3, char_4
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

		elif (arr[i] == '.' or arr[i] == '?' or arr[i] == '!') and (arr[i+1] != '<' and arr[i+1] != "'") and arr[i+2] != '/' and arr[i+3] != 's':
			char1 = arr[i+1]
			char2 = arr[i+2]
			char3 = arr[i+3]
			char4 = arr[i+4]
			char_1 = arr[i]
			char_2 = arr[i-1]
			char_3 = arr[i-2]
			char_4 = arr[i-3]
			# print "test2", arr[i], char1, char2, char3, char_1, char_2, char_3
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
			
# print pre_process

# Dump the pre_process array in a file
with open('negative_data.json', 'w') as outfile:
    json.dump(pre_process, outfile)

print len(arr)