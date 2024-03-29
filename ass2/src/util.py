import os
import numpy as np
from collections import defaultdict
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer, TfidfVectorizer

class Data:
	def __init__(self):
		train_datadir = ["./data/train/pos/", "./data/train/neg/"]
		test_datadir = ["./data/test/pos/", "./data/test/neg/"]
		self.train_corpus = []
		self.test_corpus = []
		self.Y_train = np.concatenate((np.ones((12500), dtype=int), np.zeros((12500), dtype=int)))
		self.Y_test = np.concatenate((np.ones((12500), dtype=int), np.zeros((12500), dtype=int)))
		for i in range(len(train_datadir)):
			cur_dir = train_datadir[i]
			for filename in os.listdir(cur_dir):
				with open(cur_dir+filename, 'r') as myfile:
					data=myfile.read()
					self.train_corpus.append(data)

		for i in range(len(test_datadir)):
			cur_dir = test_datadir[i]
			for filename in os.listdir(cur_dir):
				with open(cur_dir+filename, 'r') as myfile:
					data=myfile.read()
					self.test_corpus.append(data)

	def getData(self):
		return (self.train_corpus, self.test_corpus, self.Y_train, self.Y_test)
		

class NewData:
	def __init__(self):
		train_datadir = ["./data/train/pos/", "./data/train/neg/"]
		test_datadir = ["./data/test/pos/", "./data/test/neg/"]
		self.train_corpus = []
		self.test_corpus = []
		self.Y_train = np.concatenate((np.ones((12500), dtype=int), np.zeros((12500), dtype=int)))
		self.Y_test = np.concatenate((np.ones((12500), dtype=int), np.zeros((12500), dtype=int)))
		for i in range(len(train_datadir)):
			cur_dir = train_datadir[i]
			for filename in os.listdir(cur_dir):
				with open(cur_dir+filename, 'r') as myfile:
					data=myfile.read()
					self.train_corpus.append(data.split())

		for i in range(len(test_datadir)):
			cur_dir = test_datadir[i]
			for filename in os.listdir(cur_dir):
				with open(cur_dir+filename, 'r') as myfile:
					data=myfile.read()
					self.test_corpus.append(data.split())

	def getData(self):
		return (self.train_corpus, self.test_corpus, self.Y_train, self.Y_test)


class MeanEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        # if a text is empty we should return a vector of zeros
        # with the same dimensionality as all the other vectors
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        return self

    def transform(self, X):
        return np.array([
            np.mean([self.word2vec[w] for w in words if w in self.word2vec]
                    or [np.zeros(self.dim)], axis=0)
            for words in X
        ])


class TfidfEmbeddingVectorizer(object):
    def __init__(self, word2vec):
        self.word2vec = word2vec
        self.word2weight = None
        self.dim = len(word2vec.itervalues().next())

    def fit(self, X, y):
        tfidf = TfidfVectorizer(analyzer=lambda x: x)
        tfidf.fit(X)
        # if a word was never seen - it must be at least as infrequent
        # as any of the known words - so the default idf is the max of 
        # known idf's
        max_idf = max(tfidf.idf_)
        self.word2weight = defaultdict(
            lambda: max_idf,
            [(w, tfidf.idf_[i]) for w, i in tfidf.vocabulary_.items()])

        return self

    def transform(self, X):
        return np.array([
                np.mean([self.word2vec[w] * self.word2weight[w]
                         for w in words if w in self.word2vec] or
                        [np.zeros(self.dim)], axis=0)
                for words in X
            ])
