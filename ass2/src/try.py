from __future__ import division
import os
import numpy as np
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.neural_network import MLPClassifier
from util import MeanEmbeddingVectorizer
from nltk.classify import SklearnClassifier

train_datadir = ["./data/train/pos/", "./data/train/neg/"]
test_datadir = ["./data/test/pos/", "./data/test/neg/"]
train_corpus = []
test_corpus = []
Y_train = np.concatenate((np.ones((12500), dtype=int), np.zeros((12500), dtype=int)))
Y_test = np.concatenate((np.ones((12500), dtype=int), np.zeros((12500), dtype=int)))
for i in range(len(train_datadir)):
	cur_dir = train_datadir[i]
	for filename in os.listdir(cur_dir):
		with open(cur_dir+filename, 'r') as myfile:
			data=myfile.read()
			train_corpus.append(data)

for i in range(len(test_datadir)):
	cur_dir = test_datadir[i]
	for filename in os.listdir(cur_dir):
		with open(cur_dir+filename, 'r') as myfile:
			data=myfile.read()
			test_corpus.append(data)

# X_test = []
# X_train = []
# for i in range(len(train_corpus)):
# 	if i < 12500:
# 		X_train.append((train_corpus[i], '1'))
# 		X_test.append((test_corpus[i], '1'))
# 	else:
# 		X_train.append((train_corpus[i], '0'))
# 		X_test.append((test_corpus[i], '0'))

# pipeline = Pipeline([('vect', CountVectorizer(binary=True, stop_words='english')), ('clf', MultinomialNB())])
# classif = SklearnClassifier(pipeline)
# print "start"
# classif.train(X_train)
# print "end"

text_clf = Pipeline([('vect', CountVectorizer(binary=True, stop_words='english')), ('clf', MultinomialNB())])
