# -*- coding: utf-8 -*-
"""
Created on Tue Nov 19 23:51:15 2019

@author: theo local
"""
from sklearn.dummy import DummyClassifier
from sklearn.multiclass import OneVsRestClassifier
from sklearn.metrics import jaccard_score
import pyLDAvis.sklearn
import json
import numpy as np
import pandas as pd
from nltk import RegexpTokenizer
import re
import matplotlib.pyplot as plt
from rake_nltk import Rake
from nltk.tokenize import ToktokTokenizer
from string import punctuation
from sklearn.decomposition import LatentDirichletAllocation

# a stopwords list
def buildStopWords():
    fileInput = open(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\stopwords.txt", r'r', encoding='latin-1')
    for word in fileInput:
        word = word.replace("\n", "")
        spanish_stopwords.append(word)

def ifIsFromStopWords(word):
    if word in spanish_stopwords:
        return True
    return False

from stanfordcorenlp import StanfordCoreNLP
nlp = StanfordCoreNLP("http://localhost", port=9000)



def preprocess(text):
    result = ""
    for word, pos in nlp.pos_tag(text):
        if( ifIsFromStopWords(word) == False ): #sa nu fie in lista de stopwords
            result += singularizator.singularize(word) + " " #punem la forma de singular
    return result

spanish_stopwords = []
buildStopWords()
singularizator = inflector.Spanish()

regex = RegexpTokenizer(r"\b\w+\b");

def processTranscript(transcript):
    words = regex.tokenize(transcript)
    processedTranscript = ""
    for word in words:
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript

token = ToktokTokenizer()
punct = punctuation

def clean_text(text):
    ''' Lowering text and removing undesirable marks

    Parameter:
    
    text: document to be cleaned    
    '''
    
    text = text.lower()
    text = re.sub(r'[^\w\s]','',text)
    text = re.sub("(\s\d+)","",text)
    text = ' '.join( [w for w in text.split() if len(w)>3] )
    text = re.sub('\s+', ' ', text) # matches all whitespace characters
    text = text.strip(' ')
    return text

def count_tag(taggedWiki, list_words): 
    ''' Count the number of occurrences and the average score for each tag

    '''
    
    keyword_count = dict()
    
    for s in list_words: 
        keyword_count[s] = []
        keyword_count[s].append(0)
        keyword_count[s].append(0)
    
    index = 0;
    for key in taggedWiki: 
        
        if (len(taggedWiki[key]) == 0): 
            continue
        
        index += 1
        
        for s in [s for s in taggedWiki[key] if s in list_words]: 
                keyword_count[s][0] += 1
                
                    
    # conversion of our dictionary into a list
    keyword_occurences = []
    
    for tag, item in keyword_count.items():
        if(item[0] == 0):
            keyword_occurences.append([tag, item[0], 0])
        else:
            keyword_occurences.append([tag, item[0], item[0]/index])
        
    keyword_occurences.sort(key = lambda x:x[2], reverse = True)
    
    return keyword_occurences

def most_common(tags, top_tags):
    
    tags_filtered = []
    
    for tag in tags:
        
        if tag in top_tags:
            tags_filtered.append(tag)
            
    return tags_filtered

def print_top_words(model, feature_names, n_top_words, data):
    ''' It shows the top words from the different clusters of a model
    
    Parameters:
    
    model: model 
    feature_names: different words to show 
    n_top_words (int): number of words to print for each feature 
    data: data for the model
    '''

    list_topics = []
    list_occurences = []
    n_topics = model.n_components

    for i in model.transform(data):
        list_topics.append(i.argmax())
    
    for topic in range(n_topics):
        list_occurences.append(list_topics.count(topic))

    top_topics = sorted(range(len(list_occurences)), 
                        key=lambda k: list_occurences[k], reverse=True)
    
    for topic_idx, topic_id in zip(range(1, n_topics + 1), top_topics):
        message = "Tag #%d: " % topic_idx
        message += " / ".join([feature_names[i]
                             for i in model.components_[topic_id].argsort()[:-n_top_words - 1:-1]])
        print(message)
    
    print()

def lda(vectorizer, data_train, data_test):

    ''' Showing the perplexity score for several LDA models with different values
    for n_components parameter, and printing the top words for the best LDA model
    (the one with the lowest perplexity)

    Parameters:

    vectorizer: TF-IDF convertizer                                              
    data_train: data to fit the model with
    data_test: data to test
    '''

    # number of topics 
    n_top_words = 20
    best_perplexity = np.inf
    best_lda = 0
    perplexity_list = []
    n_topics_list = []
    print("Extracting term frequency features for LDA...")

    for n_topics in np.linspace(10, 50, 5, dtype='int'):
        lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(data_train)
        n_topics_list.append(n_topics)
        perplexity = lda_model.perplexity(data_test)
        perplexity_list.append(perplexity)

        # Perplexity is defined as exp(-1. * log-likelihood per word)
        # Perplexity: The smaller the better
        if perplexity <= best_perplexity:
            best_perplexity = perplexity
            best_lda = lda_model
                                
    plt.title("Evolution of perplexity score depending on number of topics")
    plt.xlabel("Number of topics")
    plt.ylabel("Perplexity")
    plt.plot(n_topics_list, perplexity_list)
    plt.show()

    print("\n The tags in the LDA model :")
    tf_feature_names = vectorizer.get_feature_names()
    print_top_words(best_lda, tf_feature_names, n_top_words, data_test)
    
def Recommend_tags_lda(text, lda_model, vectorizer_text):
    
    ''' returns up to 5 tags.
    Parameters:
    text: the transcript
    X_train: data to fit the model with
    '''
    n_topics = 10
    threshold = 0.008 #works fine with this threshold, all trancripts have tags
    list_scores = []
    list_words = []
    used = set()

    
    text_tfidf = vectorizer_text.transform([text])

    
    text_projection = lda_model.transform(text_tfidf)
    feature_names = vectorizer_text.get_feature_names()
    lda_components = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis] # normalization

    for topic in range(n_topics):
        topic_score = text_projection[0][topic]

        for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-10:-1], sorted(lda_components[topic])[:-10:-1]):
            score = word_score * topic_score

            if score >= threshold:
                list_scores.append(score)
                list_words.append(feature_names[word_idx])
                used.add(feature_names[word_idx])

    results = [tag for (y,tag) in sorted(zip(list_scores,list_words), key=lambda pair: pair[0], reverse=True)]
    
    tags = " ".join(results[:20])

    return tags


def processTranscript(transcript):
    words = regex.tokenize(transcript)
    processedTranscript = ""
    for word in words:
        word = word.lower()
        processedTranscript  = processedTranscript  + word + " "
    return processedTranscript

def avg_jaccard(y_true,y_pred):

    ''' It calculates Jaccard similarity coefficient score for each instance,and
    it finds their averange in percentage
    Parameters:
    y_true: truth labels
    y_pred: predicted labels
    '''
    jacard = np.minimum(y_true,y_pred).sum(axis=1) / np.maximum(y_true,y_pred).sum(axis=1)
    
    return jacard.mean()*100

def tags_lda_test(X_tfidf_test, X_train):
    
    ''' Recomendation system for stackoverflow posts based on a lda model, 
    it returns up to 5 tags.

    Parameters:

    X_tfidf_test: the stackoverflow posts after TF-IDF transformation
    X_train: data to fit the model with
    '''

    df_tags_test_lda = pd.DataFrame(index=[i for i in range(X_tfidf_test.shape[0])], 
             columns=['Tags_test'])
    corpus = X_tfidf_test
    list_results = []
    n_topics = 10
    threshold = 0.010

    vectorizer_text = TfidfVectorizer(analyzer='word', min_df=0.0, max_df = 1.0, 
                                    strip_accents = None, encoding = 'utf-8', 
                                    preprocessor=None, 
                                    token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                    max_features=1000)
    X_tfidf_train = vectorizer_text.fit_transform(X_train)
    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)
    corpus_projection = lda_model.transform(corpus)
    feature_names = vectorizer_text.get_feature_names()
    lda_components = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis] # normalization

    for text in range(corpus.shape[0]):
        list_scores = []
        list_words = []

        for topic in range(n_topics):
            topic_score = corpus_projection[text][topic]

            for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
                score = topic_score*word_score

                if score >= threshold:
                    list_scores.append(score)
                    list_words.append(feature_names[word_idx])

        results = [tag for (y,tag) in sorted(zip(list_scores,list_words),
                                             key=lambda pair: pair[0], reverse=True)][:5] #maximum five tags
        list_results.append(results)
    
    y_pred_lda = pd.DataFrame({'tags':list_results})

    return y_pred_lda

def Recommend_tags_lda_test(X_tfidf_test, X_train):
    
    ''' Recomendation system for upv video transcripts based on a lda model, 
    it returns up to 5 tags.

    Parameters:

    X_tfidf_test: the transcript after TF-IDF transformation
    X_train: data to fit the model with
    '''

    df_tags_test_lda = pd.DataFrame(index=[i for i in range(X_tfidf_test.shape[0])], 
             columns=['0.008', '0.009', '0.010', '0.011'])
    corpus = X_tfidf_test
    n_topics = 10

    vectorizer_text = TfidfVectorizer(analyzer='word', min_df=0.0, max_df = 1.0, 
                                    strip_accents = None, encoding = 'utf-8', 
                                    preprocessor=None, 
                                    token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                    max_features=1000)
    X_tfidf_train = vectorizer_text.fit_transform(X_train)
    lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)
    corpus_projection = lda_model.transform(corpus)
    
    feature_names = vectorizer_text.get_feature_names()
    lda_components = lda_model.components_ / lda_model.components_.sum(axis=1)[:, np.newaxis] # normalization

    for column, threshold in zip(range(4), [0.008, 0.009, 0.010, 0.011]): #  threshold to exceed to be considered as a relevant tag

        for text in range(corpus.shape[0]):
            list_scores = []
            list_words = []

            for topic in range(n_topics):
                topic_score = corpus_projection[text][topic]

                for (word_idx, word_score) in zip(lda_components[topic].argsort()[:-5:-1], sorted(lda_components[topic])[:-5:-1]):
                    score = topic_score*word_score

                    if score >= threshold:
                        list_scores.append(score)
                        list_words.append(feature_names[word_idx])

            results = [tag for (y,tag) in sorted(zip(list_scores,list_words), 
                                                 key=lambda pair: pair[0], reverse=True)]
            df_tags_test_lda.iloc[text, column] = results[:10] #maximum 10 tags

    return df_tags_test_lda

#gather in a collection (wikipediaArticles) all the data from files (col[0] are the titles of articles and col[1] are the articles theyselves)
data = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\wikipediaArticles3.0.xlsx", parse_cols = [0])
wikipediaArticles5 = data.values

wikipediaArticles = wikipediaArticles5

data = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\wikipediaArticles3.0.xlsx", parse_cols = [1])
wikipediaArticles6 = data.values

wikipediaArticles = np.concatenate((wikipediaArticles, wikipediaArticles6), axis = 0)


#clean all the wikipedia articles text. (apply some preprocessing functions to the gathered data)
index = 0
for article_list in wikipediaArticles:
    print(index)
    article_list[0] = clean_text(article_list[0])
    index += 1
    try:
        article_list[0] = preprocess(article_list[0])
    except:
        continue
    
#use rake tool for spanish
r = Rake(language="Spanish") 

tags = [] 

#extract tags for all the articles and add them in a list
for article_list in wikipediaArticles:
    r.extract_keywords_from_text(article_list[0])
    raw_rake_keywords =  r.get_ranked_phrases()
    to_add = []
    for i in range(0, 5):
        if(i < len(raw_rake_keywords)):
            preprocessed = preprocess(raw_rake_keywords[i])
            for word in preprocessed.split():
                if((len(word) > 3) and (any(char.isdigit() for char in word) == False)):
                    to_add.append(word);
    tags.append(to_add)

taggedWiki = {}

#collection with the wikipedia articles and their assigned tags with rake nltk tool
rawTaggedWiki = pd.DataFrame(columns = ["Article", "Tags"])

index = 0
for article_list in wikipediaArticles:
    taggedWiki[article_list[0]] = tags[index]
    newLine = pd.DataFrame([[article_list[0], tags[index]]],columns = ['Article', 'Tags'])
    rawTaggedWiki = rawTaggedWiki.append(newLine,ignore_index = True)
    index += 1
    
#set that will hold all the tags assigned (each of them will have a unique appeareance in this set) 
set_tags = set()

for key in taggedWiki:
    set_tags = set_tags.union(taggedWiki[key])

#saved the file with wikipedia articles labeled with rake nltk 
#so I do not have to always run the code for labelling wikipedia articles with rake
writer = pd.ExcelWriter('rawTagsWikisComplete.xlsx', engine='xlsxwriter')

rawTaggedWiki.to_excel(writer, sheet_name='Sheet1')

writer.save()

#see how many different tags we have only by using rake nltk
print(len(set_tags))

#count the occurences for every tag in the collection with all the tags
keyword_occurences = count_tag(taggedWiki, set_tags)
trunc_occurences = keyword_occurences[1:5001]
top_tags = [i[0] for i in trunc_occurences]


for key in taggedWiki:
    taggedWiki[key] = most_common(taggedWiki[key], top_tags)

print(top_tags) #keep the first 5000 most ocurred tags

relevant_data_set = {}

delete = [key for key in taggedWiki if len(taggedWiki[key]) == 0] 

for key in delete: 
    del taggedWiki[key]

print("There are " + str(len(taggedWiki)) + " labeled wikis")

#create a file with the wikipedia articles that are still labeled after deleting the tags that 
#were not in the top 5000 most occured tags
cleanTaggedWiki = pd.DataFrame(columns = ["Article", "Tags"])

for key in taggedWiki:
     newLine = pd.DataFrame([[key, taggedWiki[key]]],columns = ['Article', 'Tags'])
     cleanTaggedWiki = cleanTaggedWiki.append(newLine,ignore_index = True)

writer = pd.ExcelWriter('taggedWikisComplete.xlsx', engine='xlsxwriter')

cleanTaggedWiki.to_excel(writer, sheet_name='Sheet1')

writer.save()

#at this point we have a training data set for the lda: wikipedia articles with preprocessed tags initially obtained with rake-nltk
dataframe = pd.read_excel(r"C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\remake_all\taggedWikisComplete.xlsx")

#Sampling data set
vectorizer_X = TfidfVectorizer(analyzer='word', min_df=0.0, max_df = 1.0, 
                                   strip_accents = None, encoding = 'utf-8', 
                                   preprocessor=None, 
                                   token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                   max_features=1000)


multilabel_binarizer = MultiLabelBinarizer()
y_target = multilabel_binarizer.fit_transform(dataframe['Tags'])

X_train, X_test, y_train, y_test = train_test_split(dataframe["Article"], y_target, test_size=0.2,train_size=0.8, random_state=0)

# TF-IDF matrices
X_tfidf_train = vectorizer_X.fit_transform(X_train)
X_tfidf_test = vectorizer_X.transform(X_test)

lda(vectorizer_X, X_tfidf_train, X_tfidf_test)

best_lda = LatentDirichletAllocation(n_components=10, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)



pyLDAvis.enable_notebook()
panel = pyLDAvis.sklearn.prepare(best_lda, X_tfidf_test, vectorizer_X, mds='tsne')
panel


n_topics = 10

lda_model = LatentDirichletAllocation(n_components=n_topics, max_iter=5,
                                        learning_method='online',
                                        learning_offset=50.,
                                        random_state=0).fit(X_tfidf_train)

vectorizer_text = TfidfVectorizer(analyzer='word', min_df=0.0, max_df = 1.0, 
                                    strip_accents = None, encoding = 'utf-8', 
                                    preprocessor=None, 
                                    token_pattern=r"(?u)\S\S+", # Need to repeat token pattern
                                    max_features=1000)
vectorizer_text.fit(X_train)

df_tags_test_lda = Recommend_tags_lda_test(X_tfidf_test, X_train)

df_tags_test_lda.head(10)

median_tags = np.median(dataframe['Tags'].apply(lambda x: len(x)))
mean_tags = np.mean(dataframe['Tags'].apply(lambda x: len(x)))
print('Average number of tags in the training set: %.2f' % mean_tags)
print('Median number of tags in the training set: ', median_tags)
print('--------------------------------------')

for threshold in df_tags_test_lda.columns:
    print('Average number of tags in the test set, with a threshold of %s: %.2f' 
          % (threshold, np.mean(df_tags_test_lda[threshold].apply(lambda x: len(x)))))
    print('Median number of tags in the test set, with a threshold of %s: %d' %
          (threshold, np.median(df_tags_test_lda[threshold].apply(lambda x: len(x)))))
    print('Percentage of posts that have recommended tags in the test set, with a threshold of %s: %d' %
          (threshold, np.sum(df_tags_test_lda[threshold].apply
                             (lambda x: 1 if len(x)!=0 else 0))*100/df_tags_test_lda.shape[0]))
    print('--------------------------------------')

# With a threshold of 0.008, test set tags have a distribution that looks like train
# set tags and also 95% of the posts have recommended tags

# Dummy Classifier (baseline)

clf = OneVsRestClassifier(DummyClassifier(random_state=0))
clf.fit(X_tfidf_train, y_train)
y_pred = clf.predict(X_tfidf_test)
jaccard = avg_jaccard(y_test, y_pred)
print('Jaccard score in percentage for Dummy Classifier: %.2f ' % jaccard)

y_pred = tags_lda_test(X_tfidf_test, X_train)
y_pred = multilabel_binarizer.fit_transform(y_pred['tags'])
jaccard_score(y_test, y_pred, average='samples')
jaccard = avg_jaccard(y_test, y_pred)
print('Jaccard score in percentage for LDA recommender system: %.2f ' % jaccard)

transcriptsTagged = pd.DataFrame(columns = ["Title", "Transcript", "Tags"])

#after choosing the best lda model, we can start tagging unseen data, in other words
#the transcripts form upv

with open(r'C:\Users\theod\Desktop\Erasmus_Valencia_2019-2020\Valencia_work\ALEX\Valencia-Educ-Video\videos_upv.json',"r", encoding='utf-8') as f:
    videos_json = json.load(f)


#these two rows of code are just for testing if the model provides good tags for any given transcript
#good results are obtained for longer transcripts
text = processTranscript("¿Cómo creo una aplicación web con API de descanso?") 

print(Recommend_tags_lda(text, lda_model, vectorizer_text ))

#after testing iterate through all the transcripts and assing them tags and then save in a excel file
index3 = 0
index_json = 0
for videos in videos_json:
    
    index_json += 1
    transcript = videos["transcription"]
    print(index3)
    
    #if("metadata" not in videos):
     #   continue
    #if("keywords" not in videos["metadata"]):
     #   continue

    
    if(transcript != ""):
        
        preprocessedTranscript = clean_text(transcript)
        
        if(preprocessedTranscript != ""):
        
            keywords =  Recommend_tags_lda(preprocessedTranscript, lda_model, vectorizer_text)
            if(len(keywords) > 0 ):
                newLine = pd.DataFrame([[videos["search"]["title"],transcript, keywords]],columns = ['Title','Transcript', 'Tags'])
                transcriptsTagged = transcriptsTagged.append(newLine,ignore_index = True)
                index3 += 1

writer = pd.ExcelWriter('TaggedVideoTranscripts_AllVideos.xlsx', engine='xlsxwriter')

transcriptsTagged.to_excel(writer, sheet_name='Sheet1')

writer.save()
   
