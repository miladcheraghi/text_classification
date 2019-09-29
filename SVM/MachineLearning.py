print(__doc__)

import preprocessing
import Plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn import svm
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

print("Cleaning and parsing the training set telegram post...\n")
cleanTrain = []
train = pd.read_json('../DataSet/80_20/LabeledTrainedData.json' , encoding="utf8")

for i in range(0,len(train)):
        cleanTrain.append(preprocessing.postToWord(train["text"][i]))
        if (i % 1000 == 0):
                print("Post %d of %d...\n" % (i, len(train)))

print("**************************")
print("cleanTrain Ok...")
print("**************************")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
# vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
# vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000 , ngram_range=(2,2))
# And now testing TFIDF vectorizer:
vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

# transform training data into feature vectors
trainDataFeatures = vectorizer.fit_transform(cleanTrain)

# convert the result to an Numpy array
trainDataFeatures = trainDataFeatures.toarray()

print(trainDataFeatures.shape)

# vocab = vectorizer.get_feature_names()
# print(vocab)

# Read the test data
test = pd.read_json('../DataSet/80_20/TestData.json' , encoding="utf8")

# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
numTest = len(test["text"])
cleanTest = []

print("Cleaning and parsing the test set movie reviews...\n")
for i in range(0,numTest):
    cleanTest.append( preprocessing.postToWord(test["text"][i]) )
    if( i % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, numTest))


# Get a bag of words for the test set, and convert to a numpy array
testDataFeatures = vectorizer.transform(cleanTest)
testDataFeatures = testDataFeatures.toarray()

print("creating the Support Vector Machine...")
clf = svm.SVC(C=100.0, kernel='linear', degree=3, gamma='auto')


# How long does it take to run?
time0 = time()
print("training the Support Vector Machine...")
clf.fit(trainDataFeatures, train["class"]) 

# Use the random forest to make sentiment label predictions
# result = model.predict(testDataFeatures)
# result = text_clf.predict(testDataFeatures)
result = clf.predict(testDataFeatures)

time1 = time()
runningTime = time1 - time0

print("Result Is Ok.")

tn, fp, fn, tp = confusion_matrix(test["class"], result).ravel()
sensitivity = tp / (tp + fn)
specificity = tn / (tn + fp)

# evaluate model
print("________________________________________________")
print('AUC: %.2f' % roc_auc_score(test["class"], result))
print('Accuracy: %.2f' % accuracy_score(test["class"], result) )
print('Sensitivity: %.2f' % sensitivity )
print('Specificity: %.2f' % specificity )
print('F-measure: %.2f' % f1_score(test["class"], result, average='binary') )
print("Running Time: %.2f Seconds" % runningTime)
print("________________________________________________")

# visualize the results
Plotting.plot_confusion_matrix(test["class"], result, classes=['Political' , 'Non-political'],
                      title='Confusion matrix for SVM Classifier')

Plotting.plot_RUC(test["class"], result , title='Confusion matrix for SVM Classifier')















