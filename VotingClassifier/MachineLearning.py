print(__doc__)

import preprocessing
import Plotting
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from time import time
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
from sklearn.ensemble import VotingClassifier
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
        # print(cleanTrain[i])

print("**************************")
print("cleanTrain Ok...")
print("**************************")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
# vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000 , ngram_range=(2,2))
# And now testing TFIDF vectorizer:
#vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)


# transform training data into feature vectors
trainDataFeatures = vectorizer.fit_transform(cleanTrain)

# convert the result to an Numpy array
trainDataFeatures = trainDataFeatures.toarray()

print("train data features shape: " , trainDataFeatures.shape)


# Read the test data
test = pd.read_json('../DataSet/80_20/TestData.json' , encoding="utf8")

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

print("test data features shape: " , testDataFeatures.shape)


# Initialize Classifiers
clf1 = MultinomialNB(alpha = 0.1)
clf2 = KNeighborsClassifier(n_neighbors=3)
clf3 = svm.SVC(C=100.0, kernel='linear', degree=3, gamma='auto',probability=True)
clf4 = RandomForestClassifier(n_estimators = 70)
clf5 = DecisionTreeClassifier(max_depth=20)
clf6 = LogisticRegression()

print("Initialize a Voting Classifier classifier")
eclf = VotingClassifier(estimators=[('nb', clf1), ('knn', clf2), ('svc', clf3) , ('rf', clf4) , ( 'dt' , clf5) , ('lr', clf6)],
                                     voting='soft',weights=[2,1,1,3,2,3]) # 

# How long does it take to run?
time0 = time()

print("Training the Voting Classifier")
eclf = eclf.fit(trainDataFeatures, train["class"])

print("Use the Voting Classifier to make sentiment label predictions")
result = eclf.predict(testDataFeatures)

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
                      title='Confusion matrix for Voting Classifier')

Plotting.plot_RUC(test["class"], result , title='Confusion matrix for Voting Classifier')
