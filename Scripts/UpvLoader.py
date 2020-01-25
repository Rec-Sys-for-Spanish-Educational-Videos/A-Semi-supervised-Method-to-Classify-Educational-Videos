import pandas as pd
import json 

def loadWikipediaArticles():
    dataframe = pd.read_excel("Data/WikipediaTruthDataset.xlsx", usecols = "C")
    articles = dataframe.values

    dataframe = pd.read_excel("Data/WikipediaTruthDataset.xlsx", usecols = "D")
    keywords = dataframe.values

    dataframe = pd.read_excel("Data/WikipediaTruthDataset.xlsx", usecols = "E")
    labels = dataframe.values

    processedArticles = []

    for i in range(0,len(articles)):
        art = articles[i][0]
        art = str(art)

        if(len(art) < 50 or len(art) > 18000):
            continue

        processedArticles.append([art,str(keywords[i][0]), labels[i]])

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
            if(len(transcript) < 50 or len(transcript) > 18000):
                continue 
            upvData.append([transcript, keywords])

    return upvData

        