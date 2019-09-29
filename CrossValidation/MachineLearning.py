print(__doc__)

import preprocessing
import Plotting
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn import svm
# from sklearn.cross_validation import cross_val_score, cross_val_predict
from sklearn.model_selection import cross_val_score , cross_val_predict
from sklearn import metrics
from sklearn.metrics import roc_auc_score
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
from time import time


print("Cleaning and parsing telegram post...\n")
cleanData = []
Data = pd.read_json('../DataSet/100/Data.json' , encoding="utf8")

for i in range(0,len(Data)):
    cleanData.append(preprocessing.postToWord(Data["text"][i]))
    if (i % 1000 == 0):
        print("Post %d of %d...\n" % (i, len(Data)))

print("**************************")
print("cleanData Ok...")
print("**************************")

# Initialize the "CountVectorizer" object, which is scikit-learn's
# bag of words tool.
vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)
# vectorizer = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000 , ngram_range=(2,2))
# And now testing TFIDF vectorizer:
#vectorizer = TfidfVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,stop_words = None,max_features = 5000)


# transform training data into feature vectors
dataFeatures = vectorizer.fit_transform(cleanData)

# convert the result to an Numpy array
dataFeatures = dataFeatures.toarray()

print(dataFeatures.shape)

# vocab = vectorizer.get_feature_names()
# print(vocab)


# Initialize a Naive Bayse Classifier
# model = MultinomialNB(alpha = 0.1)
models = {
        "Naive Bayse" : MultinomialNB(alpha = 0.1),
        "KNN" : KNeighborsClassifier(n_neighbors=3), 
        "Logistic Regression" : LogisticRegression() ,
        "Decision Tree" : DecisionTreeClassifier(max_depth=20),
        "Random Forest" : RandomForestClassifier(n_estimators = 70),
        "SVM" : svm.SVC(C=100.0, kernel='linear', degree=3, gamma='auto'),        
}

for name , model in models.items():
        print(name)
        # How long does it take to run?
        time0 = time()

        scores = cross_val_score(model, dataFeatures, Data["class"], cv=10)
        print("max scores:", max(scores))

        time1 = time()
        periodOfTime = time1 - time0
        print("Running Time: %.2f Seconds" % periodOfTime)
        print("_______________________")
        


print("Result Is Ok.")

# evaluate model
# print('AUC: ', roc_auc_score(test["class"], result))
# print('Accuracy: ', accuracy_score(test["class"], result) )
# print("Running Time: %.2f Seconds" % periodOfTime)

# Plotting.plot_confusion_matrix(test["class"], result, classes=['Political' , 'Non-political'],
#                       title='Confusion matrix for Naive Bayse Classifier')

# Plotting.plot_RUC(test["class"], result , title='Confusion matrix for Naive Bayse Classifier')

# plt.show()

