import pandas as pd
import numpy as np
import preprocessing
from time import time
from sklearn.feature_extraction.text import CountVectorizer , TfidfVectorizer
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score , cross_val_predict
from scipy import stats

print("Cleaning and parsing telegram post...\n")
cleanData = []
Data = pd.read_json('../DataSet/100/Data.json' , encoding="utf8")

Data = Data[Data['class'] != 0]
# print(Data["text"][22000])
# print(Data.shape)
# input()
for i in range(0,19998): # len(Data)
    if(i == 19999):
        continue 
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

post_df = pd.DataFrame(dataFeatures)
z = np.abs(stats.zscore(post_df))

print("data shape befor remove outlier: " , post_df.shape)

threshold = 50
print(np.where(z > 50))

post_df_o = post_df[(z < 50).all(axis=1)]

print(type(post_df_o))
print("data shape after remove outlier: " , post_df_o.shape)
print(len(post_df_o))






















# outliers_fraction = 0.05
# # Define seven outlier detection tools to be compared
# classifiers = {
#         'K Nearest Neighbors (KNN)': KNN(contamination=outliers_fraction),
#         'Average KNN': KNN(method='mean',contamination=outliers_fraction)
# }

# for i, (clf_name, clf) in enumerate(classifiers.items()):
#         clf.fit(dataFeatures)
#         # predict raw anomaly score
#         scores_pred = clf.decision_function(dataFeatures) * -1

#         y_pred = clf.predict(dataFeatures)
#         n_inliers = len(y_pred) - np.count_nonzero(y_pred)
#         n_outliers = np.count_nonzero(y_pred == 1)






