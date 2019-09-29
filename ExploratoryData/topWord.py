import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
import arabic_reshaper
from bidi import algorithm as bidialg
import preprocessing

# functions ... 

def get_top_n_words(corpus, n , n_gram):
    vec = CountVectorizer(analyzer = "word",tokenizer = None,preprocessor = None,
                            stop_words = None, ngram_range=(n_gram, n_gram)).fit(corpus)
    bag_of_words = vec.transform(corpus)
    print(len(vec.get_feature_names()))
    sum_words = bag_of_words.sum(axis=0) 
    words_freq = [(word, sum_words[0, idx]) for word, idx in vec.vocabulary_.items()]
    words_freq =sorted(words_freq, key = lambda x: x[1], reverse=True)
    # myList = words_freq[4:9] + words_freq[10:14] + words_freq[15:17] + words_freq[18:27]  
    # myList = words_freq[0:14] + words_freq[16:22] 
    # return myList
    return words_freq[:n]

def persianEncoding(text):
    return bidialg.get_display( arabic_reshaper.reshape(text))

def plotBarChart(x , y , title , xlable , ylable):
    xvalues = x # tuple 
    y_pos = np.arange(len(xvalues))
    yValues = y # list

    plt.bar(y_pos, yValues, align='center', alpha=0.7)
    plt.xticks(rotation=90)
    plt.xticks(y_pos, xvalues)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(title)
    plt.show()

def main():
    # reading data 
    print("reading data")
    data = pd.read_json('../Dataset/100/Data.json' , encoding="utf8")
    
    # cleaning and parsing data
    print("cleaning and parsing data \n")
    cleanData = []
    for i in range(0,len(data)):
        cleanData.append(preprocessing.postToWord(data["text"][i]))
        if (i % 1000 == 0):
            print("Post %d of %d...\n" % (i, len(data)))
            # print(cleanTrain[i])

    # top unigrams before removing stop words
    common_words = get_top_n_words(data['text'], 20 , 1)
    df1 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df1['ReviewText'] = [persianEncoding(df1['ReviewText'][i]) for i in range(0 , len(df1['ReviewText']))]
    plotBarChart(tuple(list(df1['ReviewText'])) , list(df1['count']) , "top unigrams before removing stop words" , "Words" , "Counts" )

    # top unigrams after removing stop words
    common_words = get_top_n_words(cleanData, 40 , 1)
    df2 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df2['ReviewText'] = [persianEncoding(df2['ReviewText'][i]) for i in range(0 , len(df2['ReviewText']))]
    plotBarChart(tuple(list(df2['ReviewText'])) , list(df2['count']) , "top unigrams after removing stop words" , "Words" , "Counts" )

    # top bigrams before removing stop words
    common_words = get_top_n_words(cleanData, 20 , 2)
    df3 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df3['ReviewText'] = [persianEncoding(df3['ReviewText'][i]) for i in range(0 , len(df3['ReviewText']))]
    plotBarChart(tuple(list(df3['ReviewText'])) , list(df3['count']) , "top bigrams before removing stop words" , "Words" , "Counts" )
    
    # top bigrams after removing stop words
    common_words = get_top_n_words(cleanData, 20 , 2)
    df4 = pd.DataFrame(common_words, columns = ['ReviewText' , 'count'])
    df4['ReviewText'] = [persianEncoding(df4['ReviewText'][i]) for i in range(0 , len(df4['ReviewText']))]
    plotBarChart(tuple(list(df4['ReviewText'])) , list(df4['count']) , "top bigrams after removing stop words" , "Words" , "Counts" )


if __name__ == "__main__":
    main()















