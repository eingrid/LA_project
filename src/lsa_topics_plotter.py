import numpy as np
import json
import matplotlib.pyplot as plt
from gensim.models.coherencemodel import CoherenceModel

class Plotter:
    def __init__(self):
        pass
    
    def calculate_coherence_for_plots(self, topic_numbers: list, texts,
                                      dictionary, lsa_pipeline,
                                      preprocessed_documents_list,
                                      n_top_words,
                                      ngram_range = (1,1),
                                      filter_by_pos = [],
                                      lemmatize: bool = True,
                                      tf_idf_max_df = 0.9,
                                      tf_idf_min_df = 4,
                                      coherence_type: str = 'c_v',
                                      processes: int = 1,
                                      random_state = 0,
                                      save_name = False,
                                      verbose: int = 0):
        coherences = []
        predicted_topics_counts = []
        for topic_number in topic_numbers:
            LSAPipeline = lsa_pipeline(preprocessed_documents_list, tf_idf_max_df=tf_idf_max_df,
                                       tf_idf_min_df=tf_idf_min_df,
                                       lsa_components=topic_number, svd_n_iter=5,
                                       n_top_words=n_top_words, ngram_range=ngram_range,
                                       filter_by_pos=filter_by_pos,
                                       lemmatize=lemmatize,
                                       import_preprocessed_documents=True, random_state=random_state)
            topics = LSAPipeline.run_topics_detection()
            # tfidf_feature_names = set(LSAPipeline.tfidf_vectorizer.get_feature_names_out())
            # Calculate the coherence score using Gensim
            coherence_model = CoherenceModel(topics=topics,
                                             texts=texts,
                                             dictionary=dictionary,
                                             coherence=coherence_type,
                                             processes=processes)
            coherence_score = coherence_model.get_coherence()
            predicted_topics = LSAPipeline.transform_documents(LSAPipeline.import_documents_list)
            predicted_topics_count = [predicted_topics.count(x) for x in np.sort(np.unique(predicted_topics))]
            if verbose == 1:
                print(f'--Topic number {topic_number} coherence score: {coherence_score}')
            coherences.append(coherence_score)
            predicted_topics_counts.append(predicted_topics_count)
        if save_name:
            with open('coherences_{save_name}.json','w') as json_file:
                json.dump({'coherences': coherences,
                           'predicted_topics': predicted_topics_counts}, json_file)
        return coherences, predicted_topics_counts
    
    def plot_coherences(self, topic_numbers, coherences,
                        plot_description: str = '',
                        lang_description: str = 'Ukrainian'):
        plt.plot(topic_numbers, coherences)
        plt.title(f'Coherence for different n_topics | {plot_description} | {lang_description}')
        plt.xlabel('Topics number')
        plt.ylabel('Coherence (c_v) score')
        plt.grid()
        plt.show()