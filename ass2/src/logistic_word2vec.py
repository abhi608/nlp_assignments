from __future__ import division
import os
import numpy as np
from util import Data, MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer, NewData
from sklearn.pipeline import Pipeline
from gensim.models.word2vec import Word2Vec
from sklearn.linear_model import LogisticRegression

#---------------------------------------Word2vec-------------------------------------------------
train_corpus, test_corpus, Y_train, Y_test = NewData().getData()
model = Word2Vec(train_corpus, size=50, window=5, min_count=5, workers=2)
w2v = {w: vec for w, vec in zip(model.wv.index2word, model.wv.syn0)}
train_corpus, test_corpus, Y_train, Y_test = Data().getData()
#-------------------------MeanEmbeddingVectorizer-------------------------
text_clf = Pipeline([('vect', MeanEmbeddingVectorizer(w2v)), ('clf', LogisticRegression()),])
print "training start!"
text_clf.fit(train_corpus, Y_train)
print "training complete!"
Y_pred = text_clf.predict(test_corpus)
print "prediction complete!"
accuracy = (Y_test == Y_pred).sum() / Y_test.shape[0]
print "word2vec_MeanEmbeddingVectorizer_logistic_Accuracy: ", accuracy
#-------------------------MeanEmbeddingVectorizer-------------------------

#-------------------------TfidfEmbeddingVectorizer-------------------------
text_clf = Pipeline([('vect', TfidfEmbeddingVectorizer(w2v)), ('clf', LogisticRegression()),])
print "training start!"
text_clf.fit(train_corpus, Y_train)
print "training complete!"
Y_pred = text_clf.predict(test_corpus)
print "prediction complete!"
accuracy = (Y_test == Y_pred).sum() / Y_test.shape[0]
print "word2vec_TfidfEmbeddingVectorizer_logistic_Accuracy: ", accuracy
#-------------------------TfidfEmbeddingVectorizer-------------------------
