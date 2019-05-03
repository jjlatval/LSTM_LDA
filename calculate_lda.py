from topic_modelling import Corpus, transformations, n_topics, n_cpu_cores

# This file is used to create a beta matrix for LDA

Corpus(transf=transformations, transf_parameters={'tfidf': {'load': False, 'bzip2': True},
                                                  'lsi': {'load': False, 'topics': n_topics},
                                                  'lda': {'load': False, 'topics': n_topics, 'cpu_cores': n_cpu_cores},
                                                  'hdp': {'load': False}})
