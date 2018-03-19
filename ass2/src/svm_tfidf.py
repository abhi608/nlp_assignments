from __future__ import division
import os
import numpy as np
from util import Data
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import TfidfVectorizer

#---------------------------------------TfIdf-------------------------------------------------
train_corpus, test_corpus, Y_train, Y_test = Data().getData()
text_clf = Pipeline([('vect', TfidfVectorizer(stop_words='english')), ('clf', SVC())])
print "training start!"
text_clf.fit(train_corpus, Y_train)
print "training complete!"
Y_pred = text_clf.predict(test_corpus)
print "prediction complete!"
accuracy = (Y_test == Y_pred).sum() / Y_test.shape[0]
print "tfidf_svm_Accuracy: ", accuracy
#-------------------------------------------------------------------------------------------