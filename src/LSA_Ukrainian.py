import gzip
from io import StringIO
import pandas as pd
from datetime import datetime, timedelta
import numpy as np
import re
import string
import json
import matplotlib.pyplot as plt
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenize
from nltk import ngrams
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.decomposition import TruncatedSVD
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from gensim import matutils

import stanza
from stop_words import get_stop_words

stop_words = get_stop_words('uk')

stanza.download('uk')

stanza_pip = stanza.Pipeline(lang='uk', processors='tokenize,mwt,pos,lemma')

def tokenize_ngrams(document, n_g):
    return [' '.join(ngram) for ngram in ngrams(document.lower().split(), n_g)]

class UkrainianPreprocessor:
    def preprocess_documents(self, documents_list: list,
                             filter_by_pos: list = [], lemmatize: bool = True,
                             return_strings: bool = True,
                             verbose: int = 0):
        preprocessed_documents = []
        for i_d, doc in enumerate(documents_list):
            preprocessed_documents.append(self.preprocess(doc,
                                                          filter_by_pos=filter_by_pos,
                                                          lemmatize=lemmatize,
                                                          return_string=return_strings))
            if (verbose == 1) and ((i_d == 0) or ((i_d + 1)%100 == 0)):
                print(f'--Preprocessed documents: {i_d+1}/{len(documents_list)}')
        return preprocessed_documents
    
    # Function to preprocess documents
    def preprocess(self, document, filter_by_pos: list = [],
                   lemmatize: bool = True, return_string: bool = True):
        # Clean the document
        document = self.remove_links_content(document)
        document = self.remove_emails(document)
        document = self.remove_multiple_space(document)
        document = self.remove_hashtags(document)
        document = self.remove_punctuation(document)
        document = self.remove_multiple_space(document)
        
        # Tokenize and lemmatize
        processed_document = stanza_pip(document.lower())
        if len(filter_by_pos) > 0:
            lemmatized_words = [(word.lemma if lemmatize else word.text) for sent in processed_document.sentences for word in sent.words if (
                word.upos in filter_by_pos)]
        else:
            lemmatized_words = [(word.lemma if lemmatize else word.text) for sent in processed_document.sentences for word in sent.words]
        # Remove stopwords and punctuations
        filtered_words = [word for word in lemmatized_words if word.isalnum() and not word in stop_words]
        if return_string:
            return ' '.join(filtered_words)
        else:
            return filtered_words
    
    def remove_links_content(self, text):
        text = re.sub(r"http\S+", "", text)
        return text
    
    def remove_emails(self, text):
        return re.sub('\S+@\S*\s?', '', text)
    
    def remove_punctuation(self, text):
        """https://stackoverflow.com/a/37221663"""
        table = str.maketrans({key: None for key in string.punctuation})
        return text.translate(table)
    
    def remove_multiple_space(self, text):
        return re.sub("\s\s+", " ", text)

    def remove_hashtags(self, text):
        old_text = text + '\n'
        new_text = text
        while len(new_text) < len(old_text):
            old_text = new_text
            new_text = re.sub('(?<=[\s\n])#\S+\s*$', '', new_text)
        return new_text

class LSAPipelineUkrainian:
    def __init__(self, documents_list, tf_idf_max_df=1.0, tf_idf_min_df=1,
                 lsa_components: int = 100, svd_n_iter: int = 5,
                 n_top_words: int = 10, ngram_range: tuple = (1, 1),
                 filter_by_pos: list = [], lemmatize: bool = True,
                 random_state: int = -1, import_preprocessed_documents: bool = False,
                 verbose: int = 0):
        if not import_preprocessed_documents:
            self.import_documents_list = UkrainianPreprocessor().preprocess_documents(documents_list,
                                                                                      filter_by_pos=filter_by_pos,
                                                                                      lemmatize=lemmatize,
                                                                                      verbose=verbose)
        else:
            self.import_documents_list = documents_list
        self.tf_idf_max_df = tf_idf_max_df
        self.tf_idf_min_df = tf_idf_min_df
        self.lsa_components = lsa_components
        self.svd_n_iter = svd_n_iter
        self.n_top_words = n_top_words
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.coherence_texts_calculated = False
        self.verbose = verbose

    def run_topics_detection(self):
        tfidf_documents = self.TF_IDF()
        if self.verbose == 1:
            print('--LSA topics detection: TF-IDF calculated')
        self.TruncatedSVD(tfidf_documents)
        if self.verbose == 1:
            print('--LSA topics detection: truncated SVD calculated')
        self.topics = self.find_topics()
        return self.topics

    def transform_documents(self, new_documents_list: list):
        return list(map(np.argmax, self.svd_model.transform(
            self.tfidf_vectorizer.transform(new_documents_list))))

    def import_ready_documents(self, documents_list, texts, dictionary, corpus):
        self.import_documents_list = documents_list
        self.texts = texts
        self.dictionary = dictionary
        self.corpus = corpus
        self.coherence_texts_calculated = True

    def TF_IDF(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_df=self.tf_idf_max_df,
                                                min_df=self.tf_idf_min_df,
                                                ngram_range=self.ngram_range,
                                                stop_words=stop_words)
        return self.tfidf_vectorizer.fit_transform(self.import_documents_list)
        # tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
    
    def TruncatedSVD(self, tfidf_documents):
        if self.random_state != -1:
            self.svd_model = TruncatedSVD(n_components=self.lsa_components,
                                 n_iter=self.svd_n_iter, random_state=self.random_state)
        else:
            self.svd_model = TruncatedSVD(n_components=self.lsa_components,
                                 n_iter=self.svd_n_iter)
        self.svd_model.fit_transform(tfidf_documents)

    def find_topics(self):
        # Extract the top words for each topic
        topics = []
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.svd_model.components_):
            top_features_ind = topic.argsort()[:-self.n_top_words - 1:-1]
            top_features = [tfidf_feature_names[i] for i in top_features_ind]
            topics.append(top_features)
        return topics

    def calculate_coherence_score(self, recalculate_texts: bool = False, verbose: int = 0):
        if not self.coherence_texts_calculated or recalculate_texts:
            self.calculate_coherence_texts(verbose=verbose)
        # Calculate the coherence score using Gensim
        coherence_model = CoherenceModel(topics=self.topics,
                                         texts=self.texts,
                                         dictionary=self.dictionary,
                                         coherence='c_v')
        if verbose == 1:
            print('--Calculating the coherence score')
        self.coherence_score = coherence_model.get_coherence()
        return self.coherence_score

    def calculate_coherence_texts(self, verbose: int = 0):
        # Convert the list of top words into a list of lists of words
        tfidf_feature_names = set(self.tfidf_vectorizer.get_feature_names_out())
        if verbose == 1:
            print('--Starting forming the texts')
        self.texts = [[word for word in doc.lower().split() if (
            word in tfidf_feature_names)] for doc in self.import_documents_list]
        # Create a Gensim dictionary
        if verbose == 1:
            print('--Creating the Gensim dectionary')
        self.dictionary = Dictionary(self.texts)
        # Convert the dictionary and the corpus
        if verbose == 1:
            print('--Converting to the corpus')
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.coherence_texts_calculated = True