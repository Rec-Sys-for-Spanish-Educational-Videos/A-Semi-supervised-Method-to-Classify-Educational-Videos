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

def pipeline():

    embedders = [Transformer.BertModel(), Transformer.NnlmModel(), Transformer.UseModel() ]

    classifiers = [  RandomForestClassifier(n_estimators=50),
                     XGBClassifier(),
                     svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 5)
                  ]
    keywordsModel = svm.SVC(gamma='auto', decision_function_shape='ovo', kernel='rbf', C = 1)

    for embedder in embedders:
    
        print("Using embedder " + embedder.__class__.__name__)

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

        print("Create embeddings for co-training trainer(the embeddings for keywords will be just NNLM while the embeddings for text can vary)")
        nnlmForKeywords = Transformer.NnlmModel()
        
        print("\tNnlm wikipedia train keywords embeddings")
        nnlmWikiTrainKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(wikiTrainKeywords)

        print("\tNnlm wikipedia train validation keywords embeddings")
        nnlmWikiTrainValidationKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(wikiTrainValidationKeywords)

        print("\tNnlm wikipedia method validation keywords embeddings")
        nnlmWikiMethodValidationKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(wikiMethodValidationKeywords)

        print("\tNnlm upv keywords embeddings")
        nnlmKeywordsEmbeddings = nnlmForKeywords.createEmbeddings(preProcessedUpvData[1])

        for clf in classifiers:
            print("Using classifier " + clf.__class__.__name__)

            print("Starting the single trainer...")
            knownTranscripts, knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.singleTrainer(clf = clf, threshold = 50, upvData = [transcriptsEmbeddings, keywordsEmbeddings], wikiTrainArticles = wikiTrainArticlesEmbeddings, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                            wikiTrainValidationArticles = wikiTrainValidationArticlesEmbeddings, wikiTrainValidationKeywords = wikiTrainValidationKeywordsEmbeddings, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel())
            Method.validateTheMethod(clf, None, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticlesEmbeddings, wikiMethodValidationKeywordsEmbeddings, wikiMethodValidationLabels)


            print("Starting the cotrainer with static keywords...")
            knownTranscripts, knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.coTrainer(clf = clf, keywordsModel = keywordsModel, threshold = 50, upvData = [transcriptsEmbeddings, nnlmKeywordsEmbeddings], wikiTrainArticles = wikiTrainArticlesEmbeddings, wikiTrainKeywords = nnlmWikiTrainKeywordsEmbeddings, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                        wikiTrainValidationArticles = wikiTrainValidationArticlesEmbeddings, wikiTrainValidationKeywords = nnlmWikiTrainValidationKeywordsEmbeddings, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel(), staticKeywords= True)
            
            Method.validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticlesEmbeddings, nnlmWikiMethodValidationKeywordsEmbeddings, wikiMethodValidationLabels)

            print("Starting the cotrainer with dynamic keywords...")
            knownTranscripts,knownKeywords, knownTranscriptsLabels, unknownTranscripts, unknownKeywords = Method.coTrainer(clf = clf, keywordsModel = keywordsModel, threshold = 50, upvData = [transcriptsEmbeddings, nnlmKeywordsEmbeddings], wikiTrainArticles = wikiTrainArticlesEmbeddings, wikiTrainKeywords = nnlmWikiTrainKeywordsEmbeddings, wikiTrainLabels = wikiTrainLabels.ravel(),
                                                                                                                        wikiTrainValidationArticles = wikiTrainValidationArticlesEmbeddings, wikiTrainValidationKeywords = nnlmWikiTrainValidationKeywordsEmbeddings, wikiTrainValidationLabels = wikiTrainValidationLabels.ravel(), staticKeywords= False)
            
            Method.validateTheMethod(clf, keywordsModel, knownTranscripts, knownKeywords, knownTranscriptsLabels, wikiMethodValidationArticlesEmbeddings, nnlmWikiMethodValidationKeywordsEmbeddings, wikiMethodValidationLabels)

    

pipeline()