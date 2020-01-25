from nltk import RegexpTokenizer
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
        processedUpvTranscripts.append(processText(transcript))
        processedUpvKeywords.append(processText(keywords))

    return [processedUpvTranscripts, processedUpvKeywords]

def preprocessWikipediaData(upvData):
    processedWikipediaData = []
    processedWikipediaKeywords = []
    labels = []

    for article, keywords, label in upvData:
        processedWikipediaData.append(processText(article))
        processedWikipediaKeywords.append(processText(keywords))
        labels.append(label)

    return [processedWikipediaData,processedWikipediaKeywords, labels]


    




