import json
import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
from nltk.corpus import stopwords
import gensim as sim
import re
from sklearn.cluster import DBSCAN
from sklearn.manifold import TSNE
import matplotlib.pyplot as plt
from gensim.models import Word2Vec
import wikipedia

import tensorflow_hub as hub
import tensorflow as tf
#Extract the common words from the transcripts and the pre trained word vectorizer


dataframe = pd.read_excel("wikipediaArticles.xlsx", parse_cols = [1])

articles = dataframe.values

dataframe = pd.read_excel("wikipediaArticles.xlsx", parse_cols = [2])

labels = dataframe.values

regex = RegexpTokenizer(r"\b\w+\b");

processedArticles = []
processedLabels = []

def processTranscript(transcript):
    words = regex.tokenize(transcript)
    processedTranscript = ""
  
    for word in words:
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript


for i in range(0,len(articles)):
    art = articles[i][0]
    
    art = str(art)
    
    if len(art) < 50:
        continue
        
    processedArticles.append(processTranscript(art))
   
    processedLabels.append(labels[i])
    

from sklearn.model_selection import train_test_split

wikiTrain, wikiEvalTrain, wikiTrainTest, wikiEvalTest = train_test_split(processedArticles,processedLabels , test_size=0.30, random_state=42)

wikiEvalTrain, wikiFinalTrain, wikiEvalTest, wikiFinalTest =  train_test_split(wikiEvalTrain,wikiEvalTest , test_size=0.50, random_state=42)


print("Loading Transcripts")

transcripts = []
transcriptsKeywords = []
regex = RegexpTokenizer(r"\b\w+\b");

with open('videos_upv.json',"r", encoding='utf-8') as f:
    videos_json = json.load(f)
        

for videos in videos_json:
    transcript =  videos["transcription"]
    if("metadata" not in videos):
        continue
    if("keywords" not in videos["metadata"]):
        continue
    
    keywords_obj = videos["metadata"]["keywords"]
    keywords = ""

    if(type( keywords_obj) is list):
        for text in  keywords_obj:
            keywords+= text + " "
    else:
        keywords = keywords_obj
            
    if transcript != "" and keywords!="":
        processedTr = processTranscript(transcript)
        processedKey = processTranscript(keywords)
        
        if(processedTr !="" and processedKey!=""):
            transcripts.append(processedTr)
            transcriptsKeywords.append(processedKey)

print("Generating the vector matrix")      

embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")



X = []
Z = []
T = []
K = []
Q = []

with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    X = session.run(embed(wikiTrain))
    Z = session.run(embed(wikiEvalTrain))
    Q = session.run(embed(wikiFinalTrain))
    T = session.run(embed(transcripts))
    K = session.run(embed(transcriptsKeywords))
    
    

print("Vector matrix generated")

    
from sklearn import svm
from sklearn.model_selection import cross_val_score

wikiTrainTest = np.array(wikiTrainTest)

from collections import defaultdict

validTranscriptsIndices = defaultdict(list)
wrongTranscriptsIndices = range(0,len(transcripts))

TranscriptsX = []
TranscriptsXLabels =[]

while(True):
   
    nr = 0
    clf = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel='rbf', C=60)
    
    scores = cross_val_score(clf, X,wikiTrainTest.ravel(), cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    clf.fit(X, wikiTrainTest)
    
    result = clf.predict(Z)
    
    from sklearn.metrics import classification_report
    
    print(classification_report(wikiEvalTest, result))
    
    resultTrLabels = clf.predict(T)
    resultKeyLabels =clf.predict(K)
    
    validTranscriptsX = []
    validTranscriptsLabels = []
    
    invalidTranscriptsX = []
    invalidTranscriptsLabels = []
    invalidKeywordsX =[]
    
    wrongIndices =[]
    for i in range(0,len(resultTrLabels)):
        if(resultTrLabels[i] == resultKeyLabels[i]):
            validTranscriptsX.append(T[i]);
            TranscriptsX.append(T[i])
            TranscriptsXLabels.append(resultTrLabels[i])
            validTranscriptsLabels.append(resultTrLabels[i])
            validTranscriptsIndices[resultTrLabels[i]].append(wrongTranscriptsIndices[i])
            nr+=1
        else:
            invalidTranscriptsX.append(T[i])
            invalidTranscriptsLabels.append(resultTrLabels[i])
            invalidKeywordsX.append(K[i])
            wrongIndices.append(wrongTranscriptsIndices[i])
    
    wrongTranscriptsIndices = wrongIndices
        

    print("Valid transcripts:" + str(nr) + " / " + str(len(resultTrLabels)) )
    
    if(nr < 20):
        for j in range(0,len(invalidTranscriptsX)):
              TranscriptsX.append(invalidTranscriptsX[j])
              TranscriptsXLabels.append(invalidTranscriptsLabels[j])
    
        break
   
    X = np.append(X, np.array(validTranscriptsX),axis = 0)
    wikiTrainTest = np.append(wikiTrainTest, np.array(validTranscriptsLabels))
    
    T = np.array(invalidTranscriptsX)
    K = np.array(invalidKeywordsX)
    
    if(nr == len(resultTrLabels)):
        break
    
    
clf = svm.SVC(gamma='scale', decision_function_shape='ovo', kernel='rbf', C=60)
scores = cross_val_score(clf, np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel(), cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel())
result = clf.predict(Q)
        
print(classification_report(wikiFinalTest, result))




