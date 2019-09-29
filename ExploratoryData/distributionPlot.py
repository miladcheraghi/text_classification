import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import arabic_reshaper
from bidi import algorithm as bidialg

def plotBarChart(x , y , title , xlable , ylable):
    xvalues = x # tuple 
    y_pos = np.arange(len(xvalues))
    yValues = y # list

    plt.bar(y_pos, yValues, align='center', alpha=0.5 , color = 'blue' )
    plt.xticks(y_pos, xvalues)
    plt.xlabel(xlable)
    plt.ylabel(ylable)
    plt.title(title)
    plt.show()

def persianEncoding(text):
    return bidialg.get_display( arabic_reshaper.reshape(text))

def main():
    # train = pd.read_json('../Dataset/80_20/LabeledTrainedData.json' , encoding="utf8")
    # test = pd.read_json('../Dataset/80_20/TestData.json' , encoding="utf8")
    train = pd.read_json('../Dataset/100/Data.json' , encoding="utf8")

    x = (persianEncoding('سیاسی') , persianEncoding('غیر سیاسی'))
    y = [ train[train['class'] == 1]['class'].count() , train[train['class'] == 0]['class'].count()]
    title = 'distributions of classes'
    xlable = 'classes'
    ylable = 'counts'
    plotBarChart(x , y , title , xlable , ylable)


if __name__ == "__main__":
    main()


