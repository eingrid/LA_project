from sklearn.decomposition import NMF
from sklearn.feature_extraction.text import TfidfVectorizer
from gensim.models.coherencemodel import CoherenceModel
from gensim.corpora.dictionary import Dictionary
from multiprocessing import Pool

from src.text_preprocessor import EnglishPreprocessor
from src.NNMF import SimpleNMF
# from src.custom_nmf.nmf import NMF
from src.custom_nmf.gpt_nmf import OptimizedNMF

class NNMFPipelineEnglish:
    def __init__(self, documents_list, tf_idf_max_df=1.0, tf_idf_min_df=1,
                n_components: int = 100, max_iter: int = 5,
                n_top_words: int = 10, ngram_range: tuple = (1, 1),
                random_state: int = -1):
        self.import_documents_list = EnglishPreprocessor().preprocess_documents(documents_list)
        self.tf_idf_max_df = tf_idf_max_df
        self.tf_idf_min_df = tf_idf_min_df
        self.n_components = n_components
        self.max_iter = max_iter
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
        self.train_nmf(tfidf_documents)
        self.topics = self.find_topics()
        return self.topics

    def TF_IDF(self):
        self.tfidf_vectorizer = TfidfVectorizer(max_df=self.tf_idf_max_df,
                                                min_df=self.tf_idf_min_df,
                                                ngram_range=self.ngram_range,
                                                stop_words='english')
        return self.tfidf_vectorizer.fit_transform(self.import_documents_list)
        # tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
    
    def train_nmf(self, tfidf_documents):
        if self.random_state != -1:
            self.model = OptimizedNMF(n_components=self.n_components,
                                max_iter=self.max_iter, init='random', random_seed=self.random_state)
        else:
            self.model = OptimizedNMF(n_components=self.n_components,
                                max_iter=self.max_iter,init='random')

        self.model.fit_transform(tfidf_documents)

    def find_topics(self):
        # Extract the top words for each topic
        topics = []
        tfidf_feature_names = self.tfidf_vectorizer.get_feature_names_out()
        for topic_idx, topic in enumerate(self.model.components_):
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
        tfidf_feature_names_set = set(tfidf_feature_names)
        if verbose == 1:
            print(f'--Starting forming the texts')
        with Pool() as pool:
            self.texts = pool.starmap(preprocess_document, [(doc, tfidf_feature_names_set) for doc in self.import_documents_list])

        self.texts = [[word for word in doc.lower().split() if (
            word in tfidf_feature_names_set)] for doc in self.import_documents_list]
        # Create a Gensim dictionary
        if verbose == 1:
            print(f'--Creating the Gensim dectionary')
        self.dictionary = Dictionary(self.texts)
        # Convert the dictionary and the corpus
        if verbose == 1:
            print(f'--Converting to the corpus')
        self.corpus = [self.dictionary.doc2bow(text) for text in self.texts]
        self.coherence_texts_calculated = True

def preprocess_document(doc, tfidf_feature_names_set):
    # Lower and split the document only once, and filter using the set
    return [word for word in doc.lower().split() if word in tfidf_feature_names_set]