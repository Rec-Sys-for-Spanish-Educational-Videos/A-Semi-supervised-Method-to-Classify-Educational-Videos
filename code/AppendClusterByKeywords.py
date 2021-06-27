import json
import re
from collections import defaultdict

import inflector
import numpy as np
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from nltk import RegexpTokenizer
from sklearn import svm
from sklearn.metrics import classification_report
from sklearn.model_selection import cross_val_score, train_test_split
from stanfordcorenlp import StanfordCoreNLP

regex = RegexpTokenizer(r"\b\w+\b");

def buildStopWords():
    fileInput = open(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\helper_data\stop_words_english.txt", r'r', encoding='latin-1')
    for word in fileInput:
        word = word.replace("\n", "")
        english_stopwords.append(word)

english_stopwords = []

singularizator = inflector.English()

nlp = StanfordCoreNLP("http://localhost", port=8000)

def ifIsFromStopWords(word):
    if word in english_stopwords:
        return True
    return False

def clean_text(text):
    ''' Lowering text and removing undesirable marks

    Parameter:
    
    text: document to be cleaned    
    '''
    
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub("(\s\d+)","",text)
    text = ' '.join( [w for w in text.split() if len(w)>3] )
    text = re.sub('\s+', ' ', text) # matches all whitespace characters
    text = text.strip(' ')
    return text

def preprocess(text):
    result = ""
    text = clean_text(text)
    if len(text) > 3:
        for word, pos in nlp.pos_tag(text):
             result += singularizator.singularize(word) + " " #punem la forma de singular
    return result

def processTranscript(transcript):
    transcript = preprocess(transcript)
    words = regex.tokenize(transcript)
    processedTranscript = ""
    for word in words:
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript


# collecting wiki data
dataframe = pd.read_excel(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\data_sets\EnglishWikipediaDataSet-V1.xlsx", usecols = [1])

articles = dataframe.values

dataframe = pd.read_excel(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\data_sets\EnglishWikipediaDataSet-V1.xlsx", usecols = [2])

labels = dataframe.values

processedArticles = []
processedLabels = []

for i in range(0,len(articles)):
    art = articles[i][0]
    
    art = str(art)
    
    if len(art) < 1000:
        continue
        
    processedArticles.append(processTranscript(art))
   
    processedLabels.append(labels[i])


wikiTrain, wikiEvalTrain, wikiTrainTest, wikiEvalTest = train_test_split(processedArticles,processedLabels , test_size=0.30, random_state=42)

wikiEvalTrain, wikiFinalTrain, wikiEvalTest, wikiFinalTest =  train_test_split(wikiEvalTrain,wikiEvalTest , test_size=0.50, random_state=42)

print("Loading Transcripts")

# collecting ted data
dataframe = pd.read_csv(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\ted_main.csv", usecols = ["tags"])
tags_ted = dataframe.values

dataframe = pd.read_csv(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\ted_main.csv", usecols = ["title"])
titles_ted = dataframe.values

dataframe = pd.read_csv(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\ted_main.csv", usecols = ["description"])
descriptions_ted = dataframe.values

dataframe = pd.read_csv(r"D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\ted_main.csv", usecols = ["url"])
urls_ted = dataframe.values

# dictUrlTags is a dictionary of url string and the list with tags
dictUrlTags = {}

for i in range(0, len(urls_ted)):
    listTags = re.findall(r'\w+', tags_ted[i][0])
    dictUrlTags[str(urls_ted[i][0])] = listTags

# dictUrlTitle is a dictionary of url string and the list with words from title
dictUrlTitle = {}
for i in range(0, len(urls_ted)):
    listTitle = re.findall(r'\w+', titles_ted[i][0])
    dictUrlTitle[str(urls_ted[i][0])] = listTitle

# dictUrlDescription is a dictionary of url string and the list with words from description
dictUrlDescription = {}
for i in range(0, len(urls_ted)):
    listDescription = re.findall(r'\w+', descriptions_ted[i][0])
    dictUrlDescription[str(urls_ted[i][0])] = listDescription

with open(r'D:\lucru\Licenta Valencia based\Git repo\A-Semi-supervised-Method-to-Classify-Educational-Videos\original_dataset\transcripts.json',"r", encoding='utf-8') as f:
    videos_json = json.load(f)

dictUrlTranscript = {}

for video in videos_json:
    url = video["url"]
    transcript = video["transcript"]
    dictUrlTranscript[url] = transcript

# dictUrlTranscript is a dictionary of url string and the transcript

transcripts = []
transcriptsKeywords = []
regex = RegexpTokenizer(r"\b\w+\b");

dictIndexTranscript = {}
index = 0
for item in videos_json:
    url =  item["url"]

    if(url not in dictUrlTags):
        continue
    if(url not in dictUrlTranscript):
        continue
    
    keywords_obj = dictUrlTags[url]
    keywords = ""

    if(type(keywords_obj) is list):
        for text in  keywords_obj:
            if "TED" not in text:
                keywords+= text + " "
    else:
        keywords = ""

    if(type(dictUrlTitle[url]) is list):
        for titleWord in dictUrlTitle[url]:
            keywords += titleWord + " "

    transcript = dictUrlTranscript[url]
    if(type(dictUrlDescription[url]) is list):
        for descriptionWord in dictUrlDescription[url]:
            transcript += descriptionWord + " "

    if transcript != "" and keywords!= "":
        processedTr = processTranscript(transcript)
        processedKey = processTranscript(keywords)
        
        if(processedTr !="" and processedKey !=""):
            transcripts.append(processedTr)
            dictIndexTranscript[index] = processedTr
                
            index = index + 1
            transcriptsKeywords.append(processedKey)

print("Generating the vector matrix")      

tf.compat.v1.disable_eager_execution()
embed = hub.Module("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/1")

X = []
Z = []
T = []
K = []
Q = []


with tf.compat.v1.Session() as session:
    session.run([tf.compat.v1.global_variables_initializer(), tf.compat.v1.tables_initializer()])
    X = session.run(embed(wikiTrain))
    Z = session.run(embed(wikiEvalTrain))
    Q = session.run(embed(wikiFinalTrain))
    T = session.run(embed(transcripts))
    K = session.run(embed(transcriptsKeywords))

dictTranscriptEmbed = {}

counter = 0
for i in range(0, len(T)):
    dictTranscriptEmbed[dictIndexTranscript[counter]] = T[i]
    counter = counter + 1

wikiTrainTest = np.array(wikiTrainTest)

validTranscriptsIndices = defaultdict(list)
wrongTranscriptsIndices = range(0,len(transcripts))

TranscriptsX = []
TranscriptsXLabels =[]

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
                for k in range(0, len(dictTranscriptEmbed[key_transcript])):
                   if(dictTranscriptEmbed[key_transcript][k] != T[i][k]):
                       break
               
                    
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

clf = svm.SVC(gamma=0.001, decision_function_shape='ovo', kernel='rbf', C=60)
scores = cross_val_score(clf, np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel(), cv=10)
print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
clf.fit(np.array(TranscriptsX) ,np.array(TranscriptsXLabels).ravel())
result = clf.predict(Q)

a =1
