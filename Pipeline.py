import Scripts.UpvLoader as Loader
import Scripts.Preprocesser as Preprocesser
import Scripts.Transformer as Transformer
import Scripts.Method as Method

import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

print("Loading data...")

upvData = Loader.loadUpvData()
print("\tUpv data loaded")
wikipediaData = Loader.loadWikipediaArticles()
print("\tWikipedia data loaded")

print("Preprocessing data...")

print("\tPreprocess upv data")
preProcessedUpvData = Preprocesser.preprocessUpvData(upvData)
print("\tPreprocess wikipedia data")
preProcessedWikipediaArticles, wikipediaLabels = Preprocesser.preprocessWikipediaData(wikipediaData)


print("Split data in training and test...")
wikiTrain, wikiEvalTrain, wikiTrainLabels, wikiEvalLabels = train_test_split(preProcessedWikipediaArticles, wikipediaLabels , test_size=0.30, random_state=42)
wikiTrainValidation, wikiMethodValidation, wikiTrainValidationLabels, wikiMethodValidationLabels =  train_test_split(wikiEvalTrain,wikiEvalLabels , test_size=0.50, random_state=42)

print("Create embeddings...")

print("\tWikipedia train embeddings")
wikiTrain = Transformer.createNnlmEmbeddings(wikiTrain)
print("\tWikipedia train validation embeddings")
wikiTrainValidation = Transformer.createNnlmEmbeddings(wikiTrainValidation)
print("\tWikipedia method validation embeddings")
wikiMethodValidation = Transformer.createNnlmEmbeddings(wikiMethodValidation)

wikiTrainLabels = np.array(wikiTrainLabels)
wikiTrainValidationLabels = np.array(wikiTrainValidationLabels)
wikiMethodValidationLabels = np.array(wikiMethodValidationLabels)

transcriptsEmbeddings = Transformer.createNnlmEmbeddings(preProcessedUpvData[0])
keywordsEmbeddings = Transformer.createNnlmEmbeddings(preProcessedUpvData[1])


clf = svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 5)

print("Starting the single trainer...")
knownTranscripts, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.methodSingleTrainer(clf = clf, threshold = 20, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrain = wikiTrain, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                   wikiTrainValidation = wikiTrainValidation, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel())


print("Validating the method...")
scores = cross_val_score(clf, knownTranscripts, knownTranscriptsLabels, cv=10)
clf.fit(knownTranscripts , knownTranscriptsLabels)

result = clf.predict(wikiMethodValidation)

print("Accuracy: %0.2f (+/- %0.2f)" % (scores.mean(), scores.std()))      
print(classification_report(wikiMethodValidationLabels, result))




