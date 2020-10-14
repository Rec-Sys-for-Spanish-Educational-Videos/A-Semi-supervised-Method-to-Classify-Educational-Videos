# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:23:32 2019

@author: theo local
"""
import time
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
import gensim

import tensorflow_hub as hub
import tensorflow as tf

from rake_nltk import Rake
from nltk.tokenize import word_tokenize 
import pickle
#from spacy.lang.es.stop_words import STOP_WORDS
#from stanfordcorenlp import StanfordCoreNLP
from nltk.stem import PorterStemmer, WordNetLemmatizer
from nltk.corpus import wordnet
import nltk

regex = RegexpTokenizer(r"\b\w+\b");

def processTranscript(transcript):
    words = regex.tokenize(transcript)
    processedTranscript = ""
    for word in words: 
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript


  
#Extract the common words from the transcripts and the pre trained word vectorizer
dataframe = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\wikipediaArticles2.0.xlsx", parse_cols = [1])

articles = dataframe.values

dataframe = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\wikipediaArticles2.0.xlsx", parse_cols = [2])

labels = dataframe.values

processedArticles = []
processedLabels = []


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

dictTranscriptTitle = {}
dictTranscriptTitleGabi = {}
dictTranscriptCluster = {}
dictTranscriptEmbed = {}
dictIndexTranscript = {}
dictIndexTranscriptGabi = {}
dictTitleCluster = {}

videoIdDictionary = {}
documentsGabi = []

counter = 0;

r = Rake(language="Spanish")

i = 0

dataframe = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\remake_all\TaggedVideoTranscripts_AllVideos.xlsx", parse_cols = [2])
transcriptsData = dataframe.values

dataframe = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\remake_all\TaggedVideoTranscripts_AllVideos.xlsx", parse_cols = [3])
tags = dataframe.values

dataframe = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\remake_all\TaggedVideoTranscripts_AllVideos.xlsx", parse_cols = [1])
titles = dataframe.values

dictTranscriptTags = {}
dictTranscriptTitle = {}

index = 0
for transcript in transcriptsData:
    str_tags = re.sub(r'[^\w\s]','',tags[index][0])
    tagsCollection = str_tags.split(" ");
    dictTranscriptTags[transcript[0]] = tagsCollection
    dictTranscriptTitle[transcript[0]] = titles[index][0]
    index += 1

r = Rake("Spanish")

for transcript in dictTranscriptTags:
    #print(transcript)
    
    keywords = dictTranscriptTags[transcript]
    
    text_keywords = ""
    
    for word in keywords:
        text_keywords += word + " "

    if transcript != "" and text_keywords!= "":
        
        transcripts.append(transcript)
        dictIndexTranscript[counter] = transcript
        
        #text_keywords += dictTranscriptTitle[transcript]
        transcriptsKeywords.append(text_keywords)
    
        counter = counter + 1
        
    i = i + 1
   
        
       
    

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

counter = 0
for i in range(0, len(T)):
    dictTranscriptEmbed[dictIndexTranscript[counter]] = T[i]
    counter = counter + 1

print(counter)
print("Vector matrix generated")

from sklearn import svm
from sklearn.model_selection import cross_val_score

wikiTrainTest = np.array(wikiTrainTest)

from collections import defaultdict

validTranscriptsIndices = defaultdict(list)
wrongTranscriptsIndices = range(0,len(transcripts))

TranscriptsX = []
TranscriptsXLabels =[]

start = time.time()

while(True):
   
    nr = 0
    clf = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='rbf', C=60)
    
    #intre processed articles si labels
    scores = cross_val_score(clf, X, wikiTrainTest.ravel(), cv=10)
    print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
    #fit intre proccessed articles si labe;s
    clf.fit(X, wikiTrainTest)
    
    #prezice si pt articles de test 
    result = clf.predict(Z)
    
    from sklearn.metrics import classification_report
    
    #cat de bine prezice label pt articles de test
    print(classification_report( (np.asarray(wikiEvalTest)).ravel() , result))
    
    resultTrLabels = clf.predict(T)
    resultKeyLabels = clf.predict(K)
    
    validTranscriptsX = []
    validTranscriptsLabels = []
    
    invalidTranscriptsX = []
    invalidTranscriptsLabels = []
    invalidKeywordsX =[]
    
    wrongIndices =[]
    for i in range(0,len(resultTrLabels)):
        if(resultTrLabels[i] == resultKeyLabels[i]):
            validTranscriptsX.append(T[i])
            for key_transcript in dictTranscriptEmbed:
                ok = 1
                for k in range(0, len(dictTranscriptEmbed[key_transcript])):
                   if(dictTranscriptEmbed[key_transcript][k] != T[i][k]):
                       ok = 0
                       break
               
                if(ok == 1):
                    dictTitleCluster[dictTranscriptTitle[key_transcript]] = resultTrLabels[i]
                    
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
        print(nr)
        print(resultTrLabels)
        break




file = open("newKeywordsComplete_v2_7clusters.txt", "w", encoding="utf-8")

clf = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='rbf', C=60)
scores = cross_val_score(clf, np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel(), cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel())
result = clf.predict(Q)
        
end = time.time()
print("time: " + str(end - start))

print(classification_report((np.asarray(wikiFinalTest)).ravel(), result))

titles = []
for title_key in dictTitleCluster:
    titles.append(title_key)
  
titles.sort()  

for title_key in titles: 
    file.write(str(title_key) + " " + str(dictTitleCluster[title_key]))
    file.write('\n')
    
file.close()

filename = 'alex_modelRetrained_hisWikiNewKeywords7CLusters.sav'
pickle.dump(clf, open(filename, 'wb'))
