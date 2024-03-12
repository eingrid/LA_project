from sklearn.decomposition import TruncatedSVD
from src.text_preprocessor import EnglishPreprocessor
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary

class LSAPipelineEnglish:
    def __init__(self, documents_list, tf_idf_max_df=1.0, tf_idf_min_df=1,
                 lsa_components: int = 100, svd_n_iter: int = 5,
                 n_top_words: int = 10, ngram_range: tuple = (1, 1),
                 random_state: int = -1):
        self.import_documents_list = EnglishPreprocessor().preprocess_documents(documents_list)
        self.tf_idf_max_df = tf_idf_max_df
        self.tf_idf_min_df = tf_idf_min_df
        self.lsa_components = lsa_components
        self.svd_n_iter = svd_n_iter
        self.n_top_words = n_top_words
        self.ngram_range = ngram_range
        self.random_state = random_state
        self.coherence_texts_calculated = False

    def import_ready_documents(self, documents_list, texts, dictionary, corpus):
        self.import_documents_list = documents_list
        self.texts = texts
        self.dictionary = dictionary
        self.corpus = corpus
        self.coherence_texts_calculated = True

    def run_topics_detection(self):
        tfidf_documents = self.TF_IDF()
        self.TruncatedSVD(tfidf_documents)
        self.topics = self.find_topics()
        return self.topics

    def TF_IDF(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_df=self.tf_idf_max_df,
                                                min_df=self.tf_idf_min_df,
                                                ngram_range=self.ngram_range,
                                                stop_words='english')
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
            print(f'--Calculating the coherence score')
        self.coherence_score = coherence_model.get_coherence()
        return self.coherence_score

    def calculate_coherence_texts(self, verbose: int = 0):
        # Convert the list of top words into a list of lists of words
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        if verbose == 1:
            print(f'--Starting forming the texts')
        self.texts = [[word for word in doc.lower().split() if (
            word in tfidf_feature_names)] for doc in self.import_documents_list]
        # Create a Gensim dictionary
        if verbose == 1:
            print(f'--Creating the Gensim dectionary')
        self.dictionary = Dictionary(self.texts)
        # Convert the dictionary and the corpus
        if verbose == 1:
            print(f'--Converting to the corpus')
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.coherence_texts_calculated = True
