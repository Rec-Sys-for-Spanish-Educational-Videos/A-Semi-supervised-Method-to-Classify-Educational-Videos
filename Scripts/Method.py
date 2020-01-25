from sklearn.metrics import classification_report
from collections import defaultdict
from sklearn.model_selection import cross_val_score
from sklearn import svm
import numpy as np

def singleTrainer(clf,threshold, upvData, wikiTrainArticles, wikiTrainLabels, wikiTrainValidationArticles, wikiTrainValidationKeywords, wikiTrainValidationLabels):
    transcripts, keywords = upvData

    trainX = wikiTrainArticles
    trainY = wikiTrainLabels

    obtainedTranscripts = []
    obtainedKeywords = []
    obtainedTranscriptsLabels =[]
    iteration = 1

    while(True):
        print("Iteration #" + str(iteration) +'\n')

        numberGoodClassifiedTranscripts = 0
        
        scores = cross_val_score(clf, trainX, trainY, cv=10)
        clf.fit(trainX, trainY)
        predictedLabels = clf.predict(wikiTrainValidationArticles)

        print("10-fold cross-validation for articles model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        print("Classification report against the wiki train validation dataset for articles")
        print(classification_report(wikiTrainValidationLabels, predictedLabels))

        predictedLabels = clf.predict(wikiTrainValidationKeywords)

        print("Classification report against the wiki train validation dataset for keywords")
        print(classification_report(wikiTrainValidationLabels, predictedLabels))


        resultTranscriptsLabels = clf.predict(transcripts)
        resultKeywordsLabels = clf.predict(keywords)

        invalidTranscripts = []
        invalidKeywords = []

        for i in range(0,len(resultTranscriptsLabels)):
            if(resultTranscriptsLabels[i] == resultKeywordsLabels[i]):
                obtainedTranscripts.append(transcripts[i])
                obtainedKeywords.append(keywords[i])
                obtainedTranscriptsLabels.append(resultTranscriptsLabels[i])
                numberGoodClassifiedTranscripts+=1
            else:
                invalidTranscripts.append(transcripts[i])
                invalidKeywords.append(keywords[i])


        print("Valid transcripts:" + str(numberGoodClassifiedTranscripts) + " / " + str(len(resultTranscriptsLabels)) )

        transcripts = np.array(invalidTranscripts)
        keywords = np.array(invalidKeywords)

        trainX = np.append(wikiTrainArticles, np.array(obtainedTranscripts),axis = 0)
        trainY = np.append(wikiTrainLabels, np.array(obtainedTranscriptsLabels).ravel(), axis =  0).ravel()

        if(numberGoodClassifiedTranscripts < threshold or numberGoodClassifiedTranscripts == len(resultTranscriptsLabels)):
            return [np.array(obtainedTranscripts), np.array(obtainedKeywords), np.array(obtainedTranscriptsLabels), np.array(transcripts), np.array(keywords)]

        iteration+=1


def coTrainer(clf, keywordsModel, threshold, upvData,  wikiTrainArticles, wikiTrainKeywords, wikiTrainLabels, wikiTrainValidationArticles, wikiTrainValidationKeywords, wikiTrainValidationLabels, staticKeywords):
    transcripts, keywords = upvData

    trainA = wikiTrainArticles
    trainK = wikiTrainKeywords
    trainY = wikiTrainLabels

    obtainedTranscripts = []
    obtainedKeywords = []
    obtainedTranscriptsLabels =[]
    iteration = 1
    
    while(True):
        print("Iteration #" + str(iteration) +'\n')
    
        numberGoodClassifiedTranscripts = 0
        
        scores = cross_val_score(clf, trainA, trainY, cv=10)
        clf.fit(trainA, trainY)
        predictedLabels = clf.predict(wikiTrainValidationArticles)

        print("10-fold cross-validation for articles model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
        print("Classification report against the wiki train validation dataset for articles")
        print(classification_report(wikiTrainValidationLabels, predictedLabels))

        if(iteration == 1 or staticKeywords == False):
            scores = cross_val_score(keywordsModel, trainK, trainY, cv=10)
            keywordsModel.fit(trainK, trainY)
            predictedLabels = keywordsModel.predict( wikiTrainValidationKeywords)

            print("10-fold cross-validation for keywords model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))
            print("Classification report against the wiki train validation dataset for keywords")
            print(classification_report(wikiTrainValidationLabels, predictedLabels))

        resultTranscriptsLabels = clf.predict(transcripts)
        resultKeywordsLabels = keywordsModel.predict(keywords)

        invalidTranscripts = []
        invalidKeywords = []

        for i in range(0,len(resultTranscriptsLabels)):
            if(resultTranscriptsLabels[i] == resultKeywordsLabels[i]):
                obtainedTranscripts.append(transcripts[i])
                obtainedKeywords.append(keywords[i])
                obtainedTranscriptsLabels.append(resultTranscriptsLabels[i])
                numberGoodClassifiedTranscripts+=1
            else:
                invalidTranscripts.append(transcripts[i])
                invalidKeywords.append(keywords[i])


        print("Valid transcripts:" + str(numberGoodClassifiedTranscripts) + " / " + str(len(resultTranscriptsLabels)) )

        transcripts = np.array(invalidTranscripts)
        keywords = np.array(invalidKeywords)

        trainA = np.append(wikiTrainArticles, np.array(obtainedTranscripts),axis = 0)
        
        if(staticKeywords == False):
            trainK = np.append(wikiTrainKeywords, np.array(obtainedKeywords), axis=0)

        trainY = np.append(wikiTrainLabels, np.array(obtainedTranscriptsLabels).ravel(), axis =  0).ravel()

        if(numberGoodClassifiedTranscripts < threshold or numberGoodClassifiedTranscripts == len(resultTranscriptsLabels)):
            return [np.array(obtainedTranscripts),np.array(obtainedKeywords), np.array(obtainedTranscriptsLabels), np.array(transcripts), np.array(keywords)]
        
        iteration+=1
