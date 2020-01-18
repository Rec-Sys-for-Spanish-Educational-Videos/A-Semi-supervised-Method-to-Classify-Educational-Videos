from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.model_selection import cross_val_score
import numpy as np

def methodSingleTrainer(clf,threshold, upvData, wikiTrain, wikiTrainLabels, wikiTrainValidation, wikiTrainValidationLabels):
    transcripts, keywords = upvData

    trainX = wikiTrain
    trainY = wikiTrainLabels

    obtainedTranscripts = []
    obtainedTranscriptsLabels =[]
    iteration = 1
    while(True):
        print("Iteration #" + str(iteration) +'\n')
        iteration+=1

        numberGoodClassifiedTranscripts = 0
        
        scores = cross_val_score(clf, trainX, trainY, cv=10)
        clf.fit(trainX, trainY)
        predictedLabels = clf.predict(wikiTrainValidation)

        print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        print(classification_report(wikiTrainValidationLabels, predictedLabels))

        resultTranscriptsLabels = clf.predict(transcripts)
        resultKeywordsLabels = clf.predict(keywords)

        invalidTranscripts = []
        invalidKeywords = []

        for i in range(0,len(resultTranscriptsLabels)):
            if(resultTranscriptsLabels[i] == resultKeywordsLabels[i]):
                obtainedTranscripts.append(transcripts[i])
                obtainedTranscriptsLabels.append(resultTranscriptsLabels[i])
                numberGoodClassifiedTranscripts+=1
            else:
                invalidTranscripts.append(transcripts[i])
                invalidKeywords.append(keywords[i])


        print("Valid transcripts:" + str(numberGoodClassifiedTranscripts) + " / " + str(len(resultTranscriptsLabels)) )

        transcripts = np.array(invalidTranscripts)
        keywords = np.array(invalidKeywords)

        trainX = np.append(wikiTrain, np.array(obtainedTranscripts),axis = 0)
        trainY = np.append(wikiTrainLabels, np.array(obtainedTranscriptsLabels).ravel(), axis =  0).ravel()

        if(numberGoodClassifiedTranscripts < threshold or numberGoodClassifiedTranscripts == len(resultTranscriptsLabels)):
            return [np.array(obtainedTranscripts), np.array(obtainedTranscriptsLabels), np.array(transcripts), np.array(keywords)]