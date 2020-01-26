import Scripts.UpvLoader as Loader
import Scripts.Preprocesser as Preprocesser
import Scripts.Transformer as Transformer
import Scripts.Method as Method
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn import svm
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.model_selection import cross_val_score
from sklearn.metrics import classification_report

def pipeline(clf, keywordsModel, embedder):

    print("Using classifier " + clf.__class__.__name__ + " and embedder " + embedder.__class__.__name__)

    print("Loading data...")

    upvData = Loader.loadUpvData()
    print("\tUpv data loaded")
    wikipediaData = Loader.loadWikipediaData()
    print("\tWikipedia data loaded")

    print("Preprocessing data...")

    print("\tPreprocess upv data")
    preProcessedUpvData = Preprocesser.preprocessUpvData(upvData)
    print("\tPreprocess wikipedia data")
    preProcessedWikipediaArticles, preProcessedWikipediaKeywords, wikipediaLabels = Preprocesser.preprocessWikipediaData(wikipediaData)

    #Preprocesser.runStatistics(preProcessedUpvData, preProcessedWikipediaArticles, preProcessedWikipediaKeywords, wikipediaLabels)

    print("Split data in training and test...")

    wikiTrain, wikiEvalTrain, wikiTrainLabels, wikiEvalLabels = train_test_split(np.column_stack((np.array(preProcessedWikipediaArticles).T, np.array(preProcessedWikipediaKeywords).T)), wikipediaLabels, test_size=0.30, random_state=42)
    wikiTrainValidation, wikiMethodValidation, wikiTrainValidationLabels, wikiMethodValidationLabels =  train_test_split(wikiEvalTrain,wikiEvalLabels , test_size=0.50, random_state=42)

    wikiTrainArticles, wikiTrainKeywords = [wikiTrain[:,0:1].T.tolist()[0], wikiTrain[:,1:2].T.tolist()[0]]

    wikiTrainValidationArticles, wikiTrainValidationKeywords = [wikiTrainValidation[:, 0:1].T.tolist()[0], wikiTrainValidation[:,1:2].T.tolist()[0]]

    wikiMethodValidationArticles, wikiMethodValidationKeywords = [wikiMethodValidation[:, 0:1].T.tolist()[0], wikiMethodValidation[:,1:2].T.tolist()[0]]

    wikiTrainLabels = np.array(wikiTrainLabels)
    wikiTrainValidationLabels = np.array(wikiTrainValidationLabels)
    wikiMethodValidationLabels = np.array(wikiMethodValidationLabels)

    print("Create embeddings for single trainer(embeddings must be the same for keywords and text)")

    print("\tWikipedia train embeddings")

    print("\t\tArticles embeddings")
    wikiTrainArticlesEmbeddings = embedder.createEmbeddings(wikiTrainArticles)
    print("\t\tKeywords embeddings")
    wikiTrainKeywordsEmbeddings = embedder.createEmbeddings(wikiTrainKeywords)

    print("\tWikipedia train validation embeddings")

    print("\t\tArticles embeddings")
    wikiTrainValidationArticlesEmbeddings = embedder.createEmbeddings(wikiTrainValidationArticles)
    print("\t\tKeywords embeddings")
    wikiTrainValidationKeywordsEmbeddings = embedder.createEmbeddings(wikiTrainValidationKeywords)

    print("\tWikipedia method validation embeddings")

    print("\t\tArticles embeddings")
    wikiMethodValidationArticlesEmbeddings = embedder.createEmbeddings(wikiMethodValidationArticles)
    print("\t\tKeywords embeddings")
    wikiMethodValidationKeywordsEmbeddings = embedder.createEmbeddings(wikiMethodValidationKeywords)

    print("\tUpv embeddings")

    print("\t\tUpv transcript embeddings")
    transcriptsEmbeddings = embedder.createEmbeddings(preProcessedUpvData[0])
    print("\t\tUpv keywords embeddings")
    keywordsEmbeddings = embedder.createEmbeddings(preProcessedUpvData[1])
    
    print("Starting the single trainer...")
    knownTranscripts, knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.singleTrainer(clf = clf, threshold = 50, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticlesEmbeddings, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                    wikiTrainValidationArticles = wikiTrainValidationArticlesEmbeddings, wikiTrainValidationKeywords = wikiTrainValidationKeywordsEmbeddings, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel())
    Method.validateTheMethod(clf, None, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticlesEmbeddings, wikiMethodValidationKeywordsEmbeddings, wikiMethodValidationLabels)


    print("Create embeddings for co-training trainer(the embeddings for keywords will be just NNLM while the embeddings for text can vary)")
    nnlmForKeywords = Transformer.NnlmModel()
    
    print("\tRemake wikipedia train keywords embeddings")
    wikiTrainKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(wikiTrainKeywords)

    print("\tRemake wikipedia train validation keywords embeddings")
    wikiTrainValidationKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(wikiTrainValidationKeywords)

    print("\tRemake wikipedia method validation keywords embeddings")
    wikiMethodValidationKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(wikiMethodValidationKeywords)

    print("\tRemake upv keywords embeddings")
    keywordsEmbeddings = nnlmForKeywords.createEmbeddings(preProcessedUpvData[1])

    print("Starting the cotrainer with static keywords...")
    knownTranscripts, knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.coTrainer(clf = clf, keywordsModel = keywordsModel, threshold = 50, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticlesEmbeddings, wikiTrainKeywords = wikiTrainKeywordsEmbeddings, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                   wikiTrainValidationArticles = wikiTrainValidationArticlesEmbeddings, wikiTrainValidationKeywords = wikiTrainValidationKeywordsEmbeddings, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel(), staticKeywords= True)
    
    Method.validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticlesEmbeddings, wikiMethodValidationKeywordsEmbeddings, wikiMethodValidationLabels)

    print("Starting the cotrainer with dynamic keywords...")
    knownTranscripts,knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.coTrainer(clf = clf, keywordsModel = keywordsModel, threshold = 50, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticlesEmbeddings, wikiTrainKeywords = wikiTrainKeywordsEmbeddings, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                   wikiTrainValidationArticles = wikiTrainValidationArticlesEmbeddings, wikiTrainValidationKeywords = wikiTrainValidationKeywordsEmbeddings, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel(), staticKeywords= False)
    
    Method.validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticlesEmbeddings, wikiMethodValidationKeywordsEmbeddings, wikiMethodValidationLabels)

    


embedders = [Transformer.BertModel(), Transformer.UseModel(), Transformer.NnlmModel()]

classifiers = [    svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 5),
                   RandomForestClassifier(n_jobs=-1),
                   XGBClassifier(n_thread=-1)
              ]
keywordsModel = svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 1)

for embedder in embedders:
    for clf in classifiers:
        pipeline(clf,keywordsModel, embedder)