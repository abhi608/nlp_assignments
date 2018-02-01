from __future__ import division
import json
import pickle
import numpy as np
from sklearn import svm
from random import shuffle
from sklearn.externals import joblib

with open('positive_data.json', 'r') as f:
    data1 = json.load(f)

with open('negative_data.json', 'r') as f:
    data2 = json.load(f)

shuffle(data1)
shuffle(data2)

# print(len(data1), len(data2))

data_X_train = []
data_Y_train = []

count1 = 0
count2 = 0

while count2 < len(data2):
	data_X_train.append(data1[count1])
	data_Y_train.append(1)
	data_X_train.append(data2[count2])
	data_Y_train.append(0)
	count1 = count1 + 1
	count2 = count2 + 1

#Running this code ill make he dataset skewed
#--------------------------------------------------------------------
# while count1 < len(data1):
# 	data_X_train.append(data1[count1])
# 	data_Y_train.append(1)
# 	count1 = count1 + 1
#--------------------------------------------------------------------

X_train = np.asarray(data_X_train)
Y_train = np.asarray(data_Y_train)

sample_size = X_train.shape[0]

X_train = X_train.reshape(sample_size,-1)
X_test = X_train[550:663,:]
Y_test = Y_train[550:663]
X_train = X_train[0:550,:]
Y_train = Y_train[0:550]
# print X_train.shape
# print Y_train.shape
# print X_test.shape
# print Y_test.shape
# print X_train.shape

classifier = svm.SVC(gamma=0.05)
# print classifier

classifier.fit(X_train, Y_train)

joblib.dump(classifier, 'model.pkl')


predicted = classifier.predict(X_train)
# print predicted.shape
cur = 0
total = 0
favour = 0
for i in predicted:
	# print i
	if i == Y_train[cur]:
		favour = favour + 1
	cur = cur + 1
	total = total + 1

print "Train accuracy: " + str(favour/total)


predicted = classifier.predict(X_test)
# print predicted.shape
cur = 0
total = 0
favour = 0
for i in predicted:
	# print i
	if i == Y_test[cur]:
		favour = favour + 1
	cur = cur + 1
	total = total + 1

print "Test accuracy: " + str(favour/total)
