import pandas as pd
import numpy as np
import preprocessing
from time import time
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from scipy import stats
from sklearn.metrics import confusion_matrix
from sklearn.metrics import f1_score
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score

print("Reading Data\n")
Data = pd.read_json('../DataSet/100/Data.json' , encoding="utf8")

# Initialize the "CountVectorizer" object
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)

# threshold for z-scor
threshold = 30

politicalData = Data[Data['class'] != 0]
cleanPolitical = []

# cleaning political data 
print("cleaning political data\n")
for index in Data.index[Data['class'] != 0].tolist():
    cleanPolitical.append(preprocessing.postToWord(politicalData["text"][index]))
    if (index % 1000 == 0):
        print("Post %d ...\n" % (index))

# transform political data into feature vectors
politicalDataFeatures = vectorizer.fit_transform(cleanPolitical)
# convert the result to an Numpy array
politicalDataFeatures = politicalDataFeatures.toarray()

print("political data befor removing outlier: " , politicalDataFeatures.shape , "\n")

post_df = pd.DataFrame(politicalDataFeatures)
z = np.abs(stats.zscore(post_df))
political_o = post_df[(z < threshold).all(axis=1)]

print("political data after removing outlier: " , political_o.shape , "\n")

# ____________________________________________________________________________________________________________________ #
nonPoliticalData = Data[Data['class'] != 1]
cleanNonPolitical = []

# cleaning non political data 
print("cleaning non political data\n")
for index in Data.index[Data['class'] != 1].tolist():
    cleanNonPolitical.append(preprocessing.postToWord(nonPoliticalData["text"][index]))
    if (index % 1000 == 0):
        print("Post %d ...\n" % (index))

# transform non political data into feature vectors
nonPoliticalDataFeatures = vectorizer.fit_transform(cleanNonPolitical)
# convert the result to an Numpy array
nonPoliticalDataFeatures = nonPoliticalDataFeatures.toarray()

print("non political data befor removing outlier: " , nonPoliticalDataFeatures.shape , "\n")

post_df = pd.DataFrame(nonPoliticalDataFeatures)
z = np.abs(stats.zscore(post_df))
non_political_o = post_df[(z < threshold).all(axis=1)]

print("non political data after removing outlier: " , non_political_o.shape , "\n")

# ____________________________________________________________________________________________________________________ #
dataset = political_o[0:10000].append(non_political_o[0:10000], ignore_index = True)
# print(dataset.shape)
# print(dataset.head)
target = [1 for i in range(0,10000)] + [0 for i in range(0,10000)]
X_train, X_test, y_train, y_test = train_test_split(dataset, target, test_size=0.05, random_state=0, stratify=target)
# print(len(X_train) , len(X_test) , len(y_train) , len(y_test) , len(target))
# num1 = 0
# num0 = 0
# print(type(X_train))
# print(X_train.count(1) , X_train.count(0))
# for item in X_train:
#     print(item)
#     if(item == 1):
#         num1 = num1 + 1
#     if(item == 0):
#         num0 = num0 + 1
# print(num1 , num0)


# Initialize a Random Forest classifier with 100 trees
print("Training the random forest...")
forest = RandomForestClassifier(n_estimators = 70)

# How long does it take to run?
time0 = time()
# Fit the forest to the training set, using the bag of words as
# features and the sentiment labels as the response variable
#
# This may take a few minutes to run
forest = forest.fit( X_train, y_train )

# # Use the random forest to make sentiment label predictions
# result = forest.predict(X_test)

# time1 = time()
# runningTime = time1 - time0

# print("Result Is Ok.")

# tn, fp, fn, tp = confusion_matrix(y_test, result).ravel()
# sensitivity = tp / (tp + fn)
# specificity = tn / (tn + fp)

# # evaluate model
# print("________________________________________________")
# print('AUC: %.2f' % roc_auc_score(y_test, result))
# print('Accuracy: %.2f' % accuracy_score(y_test, result) )
# print('Sensitivity: %.2f' % sensitivity )
# print('Specificity: %.2f' % specificity )
# print('F-measure: %.2f' % f1_score(y_test, result, average='binary') )
# print("Running Time: %.2f Seconds" % runningTime)
# print("________________________________________________\n")

print("testing with another test data\n")

# Read the test data
test = pd.read_json('../DataSet/80_20/TestData.json' , encoding="utf8")
# Verify that there are 25,000 rows and 2 columns
print(test.shape)

# Create an empty list and append the clean reviews one by one
numTest = len(test["text"])
cleanTest = []

print("Cleaning and parsing the test set...\n")
for i in range(0,numTest):
    cleanTest.append( preprocessing.postToWord(test["text"][i]) )
    if( i % 1000 == 0 ):
        print("Review %d of %d\n" % (i+1, numTest))


# Get a bag of words for the test set, and convert to a numpy array
testDataFeatures = vectorizer.transform(cleanTest)
testDataFeatures = testDataFeatures.toarray()

# Use the random forest to make sentiment label predictions
result = forest.predict(testDataFeatures)

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
print("________________________________________________\n")




# for i in range(0,len(Data)):
#     cleanData.append(preprocessing.postToWord(Data["text"][i]))
#     if (i % 1000 == 0):
#         print("Post %d of %d...\n" % (i, len(Data)))

# politicalData = Data[Data['class'] != 0]

# print("**************************")
# print("cleanData Ok...")
# print("**************************")

# # Initialize the "CountVectorizer" object, which is scikit-learn's
# # bag of words tool.
# vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 500)
# # vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000 , ngram_range=(2,2))
# # And now testing TFIDF vectorizer:
# #vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)


# # transform training data into feature vectors
# dataFeatures = vectorizer.fit_transform(cleanData)

# # convert the result to an Numpy array
# dataFeatures = dataFeatures.toarray()

# print(dataFeatures.shape)

# post_df = pd.DataFrame(dataFeatures)
# z = np.abs(stats.zscore(post_df))

# print("data shape befor remove outlier: " , post_df.shape)

# threshold = 3
# print(np.where(z > 3))

# post_df_o = post_df[(z < 3).all(axis=1)]

# print(type(post_df_o))
# print("data shape after remove outlier: " , post_df_o.shape)
# print(len(post_df_o))

# model = RandomForestClassifier(n_estimators = 100)

# time0 = time()

# scores = cross_val_score(model, dataFeatures, Data["class"], cv=10)
# print("max scores:", max(scores))

# time1 = time()
# periodOfTime = time1 - time0
# print("Running Time: %.2f Seconds" % periodOfTime)
# print("_______________________")




