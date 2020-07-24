# -*- coding: utf-8 -*-
"""
Created on Wed Nov 13 00:23:32 2019

@author: theo local
"""
import json
import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
import time
import re


import tensorflow_hub as hub
import tensorflow as tf

from rake_nltk import Rake

import pickle


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

with open(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\videos_upv.json","r", encoding='utf-8') as f:
    videos_json = json.load(f)

dictTranscriptTitle = {}
#keeping the indexes corresponding to the collection formed in Gabi's algorithm for testing purposes
dictTranscriptTitleGabi = {}
dictTranscriptEmbed = {}
dictIndexTranscript = {}
dictTitleCluster = {}

videoIdDictionary = {}
documentsGabi = []

counter = 0;

r = Rake(language="Spanish")

i = 0


dictTranscriptTags = {}


for videos in videos_json:
    transcript =  videos["transcription"]
    #print(transcript)
    
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
        
    if transcript != "" and keywords!= "":
        processedTr = processTranscript(transcript)
        processedKey = processTranscript(keywords)
        
       
        if(processedTr !="" and processedKey !=""):
            transcripts.append(processedTr)
            dictIndexTranscript[counter] = processedTr
                
            counter = counter + 1
            dictTranscriptTitle[processedTr] = videos["title"]
            
            processedKey += videos["title"] + " "
            transcriptsKeywords.append(processedKey)
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
    scores = cross_val_score(clf, X,wikiTrainTest.ravel(), cv=10)
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


file = open("rakeKeywordsceva.txt", "w", encoding="utf-8")

clf = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='rbf', C=60)
scores = cross_val_score(clf, np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel(), cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel())
result = clf.predict(Q)

end = time.time()
        
print(classification_report((np.asarray(wikiFinalTest)).ravel(), result))

print(start - end)

titles = []
for title_key in dictTitleCluster:
    titles.append(title_key)
  
titles.sort()  

for title_key in titles:
    file.write(str(title_key) + " " + str(dictTitleCluster[title_key]))
    file.write('\n')
    
file.close()

ldaDataFrame = pd.DataFrame(columns = ["VideoTitle", "ProcessedTranscript", "Cluster"])

for transcriptGabi_key in dictTranscriptTitleGabi:
        if(dictTranscriptTitleGabi[transcriptGabi_key] in dictTitleCluster):
            newLine = pd.DataFrame([[dictTranscriptTitleGabi[transcriptGabi_key], transcriptGabi_key, 
                                    dictTitleCluster[dictTranscriptTitleGabi[transcriptGabi_key]]]],
                                    columns = ['VideoTitle','ProcessedTranscript', 'Cluster'])
            ldaDataFrame = ldaDataFrame.append(newLine,ignore_index = True)

writer = pd.ExcelWriter('LdaData.xlsx', engine='xlsxwriter')

#Convert the dataframe to an XlsxWriter Excel object.
ldaDataFrame.to_excel(writer, sheet_name='Sheet1')

#Close the Pandas Excel writer and output the Excel file.
writer.save()

filename = 'alex_model.sav'
pickle.dump(clf, open(filename, 'wb'))
