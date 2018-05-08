import os
import pickle

for filename in os.listdir('./data'):
	if filename.endswith(".txt"):
		with open('./data/' + filename, 'rb') as f:
  			b = pickle.loads(f.read())
  			print b