from __future__ import division
import os
import numpy as np
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from util import Data, MeanEmbeddingVectorizer, TfidfEmbeddingVectorizer

#---------------------------------------Glove-------------------------------------------------
train_corpus, test_corpus, Y_train, Y_test = Data().getData()
with open("./glove/glove.6B.50d.txt", "rb") as lines:
    w2v = {line.split()[0]: np.array(map(float, line.split()[1:])) for line in lines}

    #-------------------------MeanEmbeddingVectorizer-------------------------
    text_clf = Pipeline([('vect', MeanEmbeddingVectorizer(w2v)), ('clf', SVC()),])
    print "training start!"
    text_clf.fit(train_corpus, Y_train)
    print "training complete!"
    Y_pred = text_clf.predict(test_corpus)
    print "prediction complete!"
    accuracy = (Y_test == Y_pred).sum() / Y_test.shape[0]
    print "glove_MeanEmbeddingVectorizer_svm_Accuracy: ", accuracy
    #-------------------------MeanEmbeddingVectorizer-------------------------

    #-------------------------TfidfEmbeddingVectorizer-------------------------
    text_clf = Pipeline([('vect', TfidfEmbeddingVectorizer(w2v)), ('clf', SVC()),])
    print "training start!"
    text_clf.fit(train_corpus, Y_train)
    print "training complete!"
    Y_pred = text_clf.predict(test_corpus)
    print "prediction complete!"
    accuracy = (Y_test == Y_pred).sum() / Y_test.shape[0]
    print "glove_TfidfEmbeddingVectorizer_svm_Accuracy: ", accuracy
    #-------------------------TfidfEmbeddingVectorizer-------------------------
#-------------------------------------------------------------------------------------------
