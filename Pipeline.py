import Scripts.UpvLoader as Loader
import Scripts.Preprocesser as Preprocesser
import Scripts.Transformer as Transformer
import Scripts.Method as Method
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticles, wikiMethodValidationKeywords, wikiMethodValidationLabels):
    print("Validating the method...")

    print("Validating the transcripts")
    scores = cross_val_score(clf, knownTranscripts, knownTranscriptsLabels, cv=10)
    clf.fit(knownTranscripts, knownTranscriptsLabels)

    result = clf.predict(wikiMethodValidationArticles)

    print("10-fold cross-validation for articles model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))      
    print(classification_report(wikiMethodValidationLabels, result))

    if(keywordsModel != None):
        print("Validating the keywords")
        
        scores = cross_val_score(keywordsModel, knownKeywords, knownTranscriptsLabels, cv=10)
        keywordsModel.fit(knownKeywords, knownTranscriptsLabels)

        result = keywordsModel.predict(wikiMethodValidationKeywords)

        print("10-fold cross-validation for keywords model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))      
        print(classification_report(wikiMethodValidationLabels, result))
    else:
        print("Validating the keywords")
        
        scores = cross_val_score(clf, knownKeywords, knownTranscriptsLabels, cv=10)
        clf.fit(knownKeywords, knownTranscriptsLabels)

        result = clf.predict(wikiMethodValidationKeywords)

        print("10-fold cross-validation for keywords model accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))      
        print(classification_report(wikiMethodValidationLabels, result))


def pipeline(clf, keywordsModel, embedder):

    print("Using classifier " + clf.__class__.__name__ + " and embedder " + embedder.__class__.__name__)

    print("Loading data...")

    upvData = Loader.loadUpvData()
    print("\tUpv data loaded")
    wikipediaData = Loader.loadWikipediaArticles()
    print("\tWikipedia data loaded")

    print("Preprocessing data...")

    print("\tPreprocess upv data")
    preProcessedUpvData = Preprocesser.preprocessUpvData(upvData)
    print("\tPreprocess wikipedia data")
    preProcessedWikipediaArticles, preProcessedWikipediaKeywords, wikipediaLabels = Preprocesser.preprocessWikipediaData(wikipediaData)

    print("Split data in training and test...")

    wikiTrain, wikiEvalTrain, wikiTrainLabels, wikiEvalLabels = train_test_split(np.column_stack((np.array(preProcessedWikipediaArticles).T, np.array(preProcessedWikipediaKeywords).T)), wikipediaLabels, test_size=0.30, random_state=42)
    wikiTrainValidation, wikiMethodValidation, wikiTrainValidationLabels, wikiMethodValidationLabels =  train_test_split(wikiEvalTrain,wikiEvalLabels , test_size=0.50, random_state=42)

    wikiTrainArticles, wikiTrainKeywords = [wikiTrain[:,0:1].T.tolist()[0], wikiTrain[:,1:2].T.tolist()[0]]

    wikiTrainValidationArticles, wikiTrainValidationKeywords = [wikiTrainValidation[:, 0:1].T.tolist()[0], wikiTrainValidation[:,1:2].T.tolist()[0]]

    wikiMethodValidationArticles, wikiMethodValidationKeywords = [wikiMethodValidation[:, 0:1].T.tolist()[0], wikiMethodValidation[:,1:2].T.tolist()[0]]

    print("Create embeddings...")
    nnlmForKeywords = Transformer.NnlmModel()

    print("\tWikipedia train embeddings")
    wikiTrainArticles = embedder.createEmbeddings(wikiTrainArticles)
    wikiTrainKeywords = nnlmForKeywords.createEmbeddings(wikiTrainKeywords)

    print("\tWikipedia train validation embeddings")
    wikiTrainValidationArticles = embedder.createEmbeddings(wikiTrainValidationArticles)
    wikiTrainValidationKeywords = nnlmForKeywords.createEmbeddings(wikiTrainValidationKeywords)

    print("\tWikipedia method validation embeddings")
    wikiMethodValidationArticles = embedder.createEmbeddings(wikiMethodValidationArticles)
    wikiMethodValidationKeywords = nnlmForKeywords.createEmbeddings(wikiMethodValidationKeywords)

    wikiTrainLabels = np.array(wikiTrainLabels)
    wikiTrainValidationLabels = np.array(wikiTrainValidationLabels)
    wikiMethodValidationLabels = np.array(wikiMethodValidationLabels)

    print("\tTranscript embeddings")
    transcriptsEmbeddings = embedder.createEmbeddings(preProcessedUpvData[0])
    print("\tKeywords embeddings")
    keywordsEmbeddings = nnlmForKeywords.createEmbeddings(preProcessedUpvData[1])


    print("Starting the single trainer...")
    knownTranscripts, knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.singleTrainer(clf = clf, threshold = 1000, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticles, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                    wikiTrainValidationArticles = wikiTrainValidationArticles, wikiTrainValidationKeywords = wikiTrainValidationKeywords, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel())
    validateTheMethod(clf, None, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticles, wikiMethodValidationKeywords, wikiMethodValidationLabels)

    print("Starting the cotrainer with static keywords...")
    knownTranscripts, knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.coTrainer(clf = clf, keywordsModel = keywordsModel, threshold = 1000, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticles, wikiTrainKeywords = wikiTrainKeywords, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                   wikiTrainValidationArticles = wikiTrainValidationArticles, wikiTrainValidationKeywords = wikiTrainValidationKeywords, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel(), staticKeywords= True)
    
    validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticles, wikiMethodValidationKeywords, wikiMethodValidationLabels)

    print("Starting the cotrainer with dynamic keywords...")
    knownTranscripts,knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.coTrainer(clf = clf, keywordsModel = keywordsModel, threshold = 1000, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticles, wikiTrainKeywords = wikiTrainKeywords, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                   wikiTrainValidationArticles = wikiTrainValidationArticles, wikiTrainValidationKeywords = wikiTrainValidationKeywords, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel(), staticKeywords= False)
    
    validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticles, wikiMethodValidationKeywords, wikiMethodValidationLabels)

    


embedders = [Transformer.BertModel(), Transformer.UseModel(), Transformer.NnlmModel()]

classifiers = [    svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 5),
                   RandomForestClassifier(),
                   XGBClassifier()
              ]
keywordsModel = svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 1)

for embedder in embedders:
    for clf in classifiers:
        pipeline(clf,keywordsModel, embedder)