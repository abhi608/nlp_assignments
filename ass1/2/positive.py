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

tmp = ['[',' ','.',']','/','<','>',',','\n',';',':','-','?',"'",')','(','!','*']

for c in tmp:
	one_hot[c] = count
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

# print arr

# fill the pre_process array
for i,_ in enumerate(arr):
	# print c, i
	if i < len(arr)-7:
		# print "test", i, arr[i]
		if arr[i] == '<' and arr[i+1] == '/' and arr[i+2] == 's' and arr[i+3] == '>': # If we found a sentence terminator
			cur = 7;
			# store the characters in left and right window(current window size is 3)
			if arr[i+cur] != ' ' and arr[i+cur] != '\n':
				char1 = ' '
				char2 = arr[i+cur]
				char3 = arr[i+cur+1]
			else:
				char1 = arr[i+cur]
				char2 = arr[i+cur+1]
				char3 = arr[i+cur+2]
			char_1 = arr[i-1]
			char_2 = arr[i-2]
			char_3 = arr[i-3]
			# print cur, arr[i], char1, char2, char3, char_1, char_2, char_3
			
			#Vectorize the current window
			obj = []
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

			# store the current data point in the pre_process array
			pre_process.append(obj)
			
print pre_process

# Dump the pre_process array in a file
with open('positive_data.json', 'w') as outfile:
    json.dump(pre_process, outfile)

print len(arr)