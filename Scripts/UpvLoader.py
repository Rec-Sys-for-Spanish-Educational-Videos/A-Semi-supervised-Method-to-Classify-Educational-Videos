import pandas as pd
import json 

def loadWikipediaData():
    dataframe = pd.read_excel("Data/WikipediaTruthDataset.xlsx", usecols = "C")
    articles = dataframe.values

    dataframe = pd.read_excel("Data/WikipediaTruthDataset.xlsx", usecols = "D")
    keywords = dataframe.values

    dataframe = pd.read_excel("Data/WikipediaTruthDataset.xlsx", usecols = "E")
    labels = dataframe.values

    processedArticles = []

    for i in range(0,len(articles)):
        label = str(labels[i][0])
        label = label[label.find("\'")+1 : label.find("\'")+2]

        processedArticles.append([str(articles[i][0]),str(keywords[i][0]),int(label)])

    return processedArticles


def loadUpvData():
    with open('Data/videos_upv.json',"r", encoding='utf-8') as f:
        videos_json = json.load(f)

    upvData = []

    for videos in videos_json:
        transcript =  videos["transcription"]
        if("metadata" not in videos):
            continue
        if("keywords" not in videos["metadata"]):
            continue
        
        keywords_obj = videos["metadata"]["keywords"]
        keywords = ""

        if(type(keywords_obj) is list):
            for text in  keywords_obj:
                keywords+= text + " "
        else:
            keywords = keywords_obj
                
        if transcript != "" and keywords!="":
            upvData.append([transcript, keywords])

    return upvData

        