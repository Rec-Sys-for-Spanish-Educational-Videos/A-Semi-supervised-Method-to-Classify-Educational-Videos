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


for i in range(0,len(articles)):
    art = articles[i][0]
    
    art = str(art)
    
    if len(art) < 50:
        continue
   
    art = art.lower()
    
    words = regex.tokenize(art)
    
  
    
    processedArticle = ""
    
    for word in words:
        processedArticle+= word + " "
        
    processedArticles.append(processedArticle)
   
    processedLabels.append(labels[i])
    


def processTranscript(transcript):
    words = regex.tokenize(transcript)
    processedTranscript = ""
  
    for word in words:
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript




from sklearn.model_selection import train_test_split

wikiTrain, wikiEvalTrain, wikiTrainTest, wikiEvalTest = train_test_split(processedArticles,processedLabels , test_size=0.10, random_state=42)


print("Load pretrained model for word2vec")

transcripts = []

regex = RegexpTokenizer(r"\b\w+\b");

with open('videos_upv.json',"r", encoding='utf-8') as f:
    data_json = json.load(f)
for video in data_json:
    transcript =  video["transcription"]
    if transcript != "" and len(transcript) >= 6581:
        transcripts.append(processTranscript(transcript))
        

print("Generating the vector matrix")      

embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")


X = []
Z = []
T = []
with tf.Session() as session:
    session.run([tf.global_variables_initializer(), tf.tables_initializer()])
    X = session.run(embed(wikiTrain))
    Z = session.run(embed(wikiEvalTrain))
    T = session.run(embed(transcripts))
    

print("Vector matrix generated")
'''
print("Reducing to 2 dimensions")
X_embedded = TSNE(n_components = 2, verbose = 1).fit_transform(X)

print("Done")

plt.scatter(X_embedded[:,0:1].ravel(), X_embedded[:,1:2].ravel(), c = processedLabels.ravel())
plt.show()
'''
    
from sklearn import svm
from sklearn.model_selection import cross_val_score

clf = svm.SVC(gamma = 'auto', decision_function_shape='ovo', kernel='rbf', C=10)
scores = cross_val_score(clf, X, np.array(wikiTrainTest).ravel(), cv=20)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(X, wikiTrainTest)

result = clf.predict(T)

from collections import defaultdict

d = defaultdict(list)

for i in range(0,len(result)):
    d[result[i]].append(i)


clf = svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C=20)

scores = cross_val_score(clf, T, result, cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std() * 2))
clf.fit(T,result)

result = clf.predict(Z)

from sklearn.metrics import classification_report

print(classification_report(wikiEvalTest, result))
