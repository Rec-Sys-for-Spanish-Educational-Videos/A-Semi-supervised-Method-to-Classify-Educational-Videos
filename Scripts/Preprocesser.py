from nltk import RegexpTokenizer
from itertools import groupby
from collections import defaultdict
import matplotlib.pyplot as plt
regexWords = RegexpTokenizer(r"\b\w+\b")

def processText(text):
    words = regexWords.tokenize(text)
    processedText = ""
  
    for word in words:
        word = word.lower()
        processedText  = processedText  + word + " "

    return processedText


def preprocessUpvData(upvData):
    
    processedUpvTranscripts = []
    processedUpvKeywords = []

    for transcript, keywords in upvData:
        processedText = processText(transcript)
        processedKeywords = processText(keywords)

        words = processedText.split()
        if(len(words) < 10 or len(words) > 2800):
            continue

        processedUpvTranscripts.append(processedText)
        processedUpvKeywords.append(processedKeywords)

    return [processedUpvTranscripts, processedUpvKeywords]

def preprocessWikipediaData(wikipediaData):
    processedWikipediaData = []
    processedWikipediaKeywords = []
    labels = []

    for article, keywords, label in wikipediaData:
        processedText = processText(article)
        processedKeywords = processText(keywords)

        words = processedText.split()
        if(len(words) < 10 or len(words) > 2800):
            continue

        processedWikipediaData.append(processedText)
        processedWikipediaKeywords.append(processedKeywords)
        labels.append(label)

    return [processedWikipediaData,processedWikipediaKeywords, labels]

def runStatistics(preProcessedUpvData, preProcessedWikipediaArticles, preProcessedWikipediaKeywords, wikipediaLabels):

      plt.boxplot([[len(transcript.split()) for transcript in preProcessedUpvData[0]],
                  [len(article.split()) for article in preProcessedWikipediaArticles]],labels=['transcripts','wikipedia articles'])
      plt.ylabel("no. words")

      plt.show()

      plt.boxplot([[len(keywords.split()) for keywords in preProcessedUpvData[1]],
                  [len(keywords.split()) for keywords in preProcessedWikipediaKeywords]],labels=['transcripts keywords', 'wikipedia keywords'])
      plt.ylabel("no. words")

      plt.show()

      plt.hist([len(transcript.split()) for transcript in preProcessedUpvData[0]],bins=200)
      plt.xlabel("no. words")
      plt.ylabel("number transcripts")

      plt.show()

      plt.hist([len(article.split()) for article in preProcessedWikipediaArticles],bins=200)
      plt.xlabel("no. words")
      plt.ylabel("number wikipedia articles")

      plt.show()

      plt.hist([len(keywords.split()) for keywords in preProcessedUpvData[1]],bins=200)
      plt.xlabel("no. words")
      plt.ylabel("number transcript keywords")

      plt.show()

      plt.hist([len(keywords.split()) for keywords in preProcessedWikipediaKeywords],bins=200)
      plt.xlabel("no. words")
      plt.ylabel("number wikipedia keywords")

      plt.show()

      freq= defaultdict( int )
      for label in wikipediaLabels:
            freq[label] += 1

      plt.bar([0,5,10], height=[freq[0], freq[1],freq[2]])
      plt.xticks([0,5,10], ['biology', 'engineering', 'humanities'])
      plt.xlabel("")
      plt.ylabel("number wikipedia articles")

      plt.show()

     



    




