# This file is used to call the topic modelling object

import os.path
import logging
from gensim import corpora, models
import smart_open
import re

logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
DOCUMENT_NAME = 'alice.txt'
DOCUMENT_PATH = './data/'
# Do not rename calculated files!

n_topics = 100
n_cpu_cores = 16

dataset_specific_stoplist = " new bank year name time another three see general m. + s. p. part sq. . e. c. a. upon" \
                            " thus like however small large several deg. case second found known became found" \
                            " among much i. far (see v. h. ft. fig. j. ii. iii. iv. vi. however, still which, f. w." \
                            " even d. vol. without lat. l. given every thought total act ft. ft.) (from @ ft.; (1900)" \
                            " lat. pop. ft.), ``the '' is, was, lat., @, city, city. pp. (q.v.) -- well great | b.c., " \
                            " b.c. 4th 14th = a.d. alps. vols. vols., 15th (gr. (2 m.), m.) (new ix. v. vii. viii. n.e." \
                            " put n. / no. (fig. ed. e.g. i.e. ie xxi. sec. sec (cf. vi. ft. ft all. all end v., i.," \
                            " ii., iii., iv., v., vi., vii., viii., ix., x., xi., xii., xiii., & &c &c. ff. ff b.c.)," \
                            " b.c.) prof. (d. (fr. (q.v..) .. . w.s.w. , : ; ``no s.w. end. i., -> (j. ital. (for &c.," \
                            " k. soc. journ. d'un w. in. n.) a, w.) c.) cf. (known with, s.v. on. (l. it, e, m, b, mss." \
                            " (born 2). (q.v.), '\' 3rd 1st 2nd take sure go look come got went nothing thing never" \
                            " ever get took gave little mock just rather shall quite looked looking began think tell" \
                            " long good might find back done way oh"

transformations = ['lda']
# transformations = ['tfidf', 'lsi', 'lda', 'hdp']
transformation_parameters = {'tfidf': {'load': False, 'bzip2': True}, 'lsi': {'load': False, 'topics': n_topics},
                             'lda': {'load': True, 'topics': n_topics, 'cpu_cores': n_cpu_cores}, 'hdp': {'load': False}}


# Corpus contains basic functions for building and manipulating corpora, see comments for more detail
class Corpus(object):
    def __init__(self, corpus_format='mm', override_dictionary=False, override_corpus=False, transf=transformations,
                 transf_parameters=transformation_parameters, document_name=DOCUMENT_PATH+DOCUMENT_NAME):
        if transf is None:  # tests dictionary contains desired tests and their corresponding parameters
            transf = []  # check #### transformations #### section
            #  for information on how to access different tests
        if transf_parameters is None:
            transf_parameters = dict()
        self.transformations = transf
        self.transformation_parameters = transf_parameters
        self.tfidf = None
        self.lsi = None
        self.lda = None
        self.hdp = None
        self.document = document_name
        self.filename = os.path.splitext(self.document)[0].split('/')[-1]
        self.override_corpus = override_corpus
        stoplist = "a about above after again against all also am an and any are aren't as at be because been before " \
                   "being below between both but by can can't cannot could couldn't did didn't do does doesn't doing don't " \
                   "down during each few first for from further had hadn't has hasn't have haven't having he he'd he'll he's " \
                   "her here here's hers herself him himself his how how's i i'd i'll i'm i've if in into is isn't it " \
                   "it's its itself last let's made make many may me more most mustn't my myself no nor not now of off on one once only or other ought " \
                   "our ours ourselves out over own said  say same shan't' she she'd she'll she's should shouldn't so some such " \
                   "than that that's the their theirs them themselves then there there's these they they'd they'll " \
                   "they're they've this those through to too two under until up us very was wasn't we we'd we'll we're we've " \
                   "were weren't what what's when when's where where's which while who who's whom why why's will with won't " \
                   "would wouldn't you you'd you'll you're you've your yours yourself yourselves"
        stoplist += dataset_specific_stoplist
        self.stoplist = set(stoplist.split())  # remove common words

        self.tokens = list()

        # tokenize text
        with smart_open.file_smart_open(self.document) as text:
            for line in text:
                self.tokens.append([re.sub('[^a-zA-Z0-9]', '', i) for i in line.lower().split()
                                    if re.sub('[^a-zA-Z0-9]', '', i) not in stoplist])

        self.dictionary = corpora.Dictionary(self.tokens)
        self.dictionary.filter_extremes(no_below=2, no_above=100)  # filters out words that appear rarely or too often
        self.corpus_format = corpus_format
        self.corpus = self.load_corpus(override_corpus=override_corpus)  # corpus holds a bag of words tuple
        if override_dictionary is True:
            self.save_dictionary()
        self.go_through_transformations()

    # Memory-friendly approach where only 1 vector is present in RAM at once
    def __iter__(self):
        for line in open(self.document):
            yield self.dictionary.doc2bow(line.lower().split())

    # Save & load functions
    def save_dictionary(self):
        self.dictionary.save(DOCUMENT_PATH + self.filename + '.dict')

    def load_dictionary(self):
        return corpora.Dictionary.load(DOCUMENT_PATH + self.filename + '.dict')

    def save_corpus(self, c):
        if c:
            c.serialize(DOCUMENT_PATH + self.filename + '_corpus.' + self.corpus_format, self)
        else:
            return 'Give me a corpus to save'

    def load_corpus(self, override_corpus=True):
        if self.corpus_format == 'svmlight':  # Joachim's SVMlight format
            try:
                c = corpora.SvmLightCorpus(fname=(DOCUMENT_PATH + self.filename + '_corpus.' + self.corpus_format))
                if override_corpus is True:
                    self.save_corpus(c)
            except:
                c = corpora.SvmLightCorpus
                self.save_corpus(c)
        elif self.corpus_format == 'lda-c':  # Blei's LDA-C format
            try:
                c = corpora.BleiCorpus(fname=(DOCUMENT_PATH + self.filename + '_corpus.' + self.corpus_format))
                if override_corpus is True:
                    self.save_corpus(c)
            except:
                c = corpora.BleiCorpus
                self.save_corpus(c)
        elif self.corpus_format == 'low':  # GibbsLDA++ format
            try:
                c = corpora.LowCorpus(fname=(DOCUMENT_PATH + self.filename + '_corpus.' + self.corpus_format))
                if override_corpus is True:
                    self.save_corpus(c)
            except:
                c = corpora.LowCorpus
                self.save_corpus(c)
        else:  # Default Market Matrix format
            try:
                c = corpora.MmCorpus(fname=(DOCUMENT_PATH + self.filename + '_corpus.' + self.corpus_format))
                if override_corpus is True:
                    self.save_corpus(c)
            except:
                c = corpora.MmCorpus
                self.save_corpus(c)
        return c

    # Transformations

    def go_through_transformations(self):
        for t in self.transformations:
            params = self.transformation_parameters.get(t, None)
            if params is None:
                return 'There was an error with accessing %s test parameters' % t
            if t == 'tfidf':
                if params.get('load', None) is False:
                    self.tfidf = self.calculate_tfidf()
                else:
                    self.tfidf = self.load_calculated_tfidf()
            if t == 'lsi':
                if params.get('load', None) is False:
                    self.lsi = self.calculate_lsi(params=params)
                else:
                    self.lsi = self.load_calculated_lsi()
            if t == 'lda':
                if params.get('load', None) is False:
                    self.lda = self.calculate_lda(params=params)
                else:
                   self.lda = self.load_calculated_lda()
            if t == 'hdp':
                if params.get('load', None) is False:
                    self.hdp = self.calculate_hdp()
                else:
                    self.hdp = self.load_calculated_hdp()

    # Term frequency-inverse document frequency
    def calculate_tfidf(self):
        tfidf = models.TfidfModel(self)
        tfidf.save(DOCUMENT_PATH + self.filename + '.tfidf')
        return tfidf

    def load_calculated_tfidf(self):
        return models.TfidfModel.load(DOCUMENT_PATH + self.filename + '.tfidf')

    # Latent Semantic Indexing
    def calculate_lsi(self, params):
        if self.tfidf is None:
            self.tfidf = self.tfidf()
        lsi = models.LsiModel(self.tfidf[self], id2word=self.dictionary, num_topics=params.get('topics', n_topics))
        lsi.save(DOCUMENT_PATH + self.filename + '.lsi')
        return lsi

    def load_calculated_lsi(self):
        return models.LsiModel.load(DOCUMENT_PATH + self.filename + '.lsi')

    # Latent Dirichlet Allocation
    def calculate_lda(self, params):
        if params.get('cpu_cores', 1) > 1:
            lda = models.LdaMulticore(self, id2word=self.dictionary, num_topics=params.get('topics', n_topics),
                                      workers=params.get('cpu_cores', params.get('cpu_cores', n_cpu_cores)))
        else:
            lda = models.LdaModel(self, id2word=self.dictionary, num_topics=params.get('topics', n_topics))
        lda.save(DOCUMENT_PATH + self.filename + '.lda')
        return lda

    def load_calculated_lda(self):
        return models.LdaModel.load(DOCUMENT_PATH + self.filename + '.lda')

    # Hierarchical Dirichlet Process
    # HDP is a non-parametric Bayesian method but it is still 'rough around the edges', so use it with care
    def calculate_hdp(self):
        hdp = models.HdpModel(self.corpus, id2word=self.dictionary)
        hdp.save(DOCUMENT_PATH + self.filename + '.hdp')
        return hdp

    def load_calculated_hdp(self):
        return models.HdpModel.load(DOCUMENT_PATH + self.filename + '.hdp')

    def get_word_ids_by_topic(self, topic_id):
        word_vec = list()
        for word_id in self.lda.get_topic_terms(topic_id):
            word_vec.append(word_id)
        return word_vec

    def get_words_by_topic(self, topic_id):
        word_id_vec = self.get_word_ids_by_topic(topic_id)
        word_vec = word_id_vec
        for i in range(0, len(word_id_vec)):
            try:
                word_vec[i] = (self.dictionary.get(int(word_id_vec[i][0])),
                               word_id_vec[i][1])
            except:
                pass
        return word_vec