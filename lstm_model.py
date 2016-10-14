import theano as theano
import theano.tensor as T
from datetime import datetime
import sys
import smart_open
import nltk
import re
from tqdm import tqdm
import itertools
from utils import *
import os
from topic_modelling import Corpus, n_topics
import random

VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '4000'))
HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '128'))
LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))
NEPOCH = int(os.environ.get('NEPOCH', '1001'))

FILE = './data/alice.txt'
MODEL_FILE = './trained_network_128_dim/alice-lstm-epoch99.npz'

vocabulary_size = VOCABULARY_SIZE
unknown_token = "UNKNOWN_TOKEN"
sentence_start_token = "SENTENCE_START"
sentence_end_token = "SENTENCE_END"


class LSTMTheano:
    
    def __init__(self, word_dim, hidden_dim=128, n_topics=n_topics, topic_matrix=None, bptt_truncate=-1):
        # Assign instance variables
        self.word_dim = word_dim
        self.hidden_dim = hidden_dim
        self.n_topics = n_topics
        self.bptt_truncate = bptt_truncate

        # Initialize the network parameters

        E = np.random.uniform(-np.sqrt(1./word_dim), np.sqrt(1./word_dim), (hidden_dim, word_dim))
        U = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        W = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (6, hidden_dim, hidden_dim))
        V = np.random.uniform(-np.sqrt(1./hidden_dim), np.sqrt(1./hidden_dim), (word_dim, hidden_dim))
        b = np.zeros((6, hidden_dim))
        c = np.zeros(word_dim)

        # Context matrices
        F = np.random.uniform(-np.sqrt(1./n_topics), np.sqrt(1./n_topics), (word_dim, n_topics))
        G = np.random.uniform(-np.sqrt(1./n_topics), np.sqrt(1./n_topics), (word_dim, n_topics))

        # Context vectors
        if topic_matrix is not None:
            self.topic_matrix = topic_matrix
            f = topic_matrix[:, random.randint(0, n_topics - 1)]  # Take a random topic from beta matrix
        else:
            f = np.zeros(n_topics)
            f.fill(0.000001)  # Add a small constant

        f = f / np.linalg.norm(f)  # Normalize f
        t = f  # t vector takes in the first selected topic as initial input


        # Theano: Created shared variables
        self.E = theano.shared(name='E', value=E.astype(theano.config.floatX))
        self.U = theano.shared(name='U', value=U.astype(theano.config.floatX))
        self.W = theano.shared(name='W', value=W.astype(theano.config.floatX))
        self.V = theano.shared(name='V', value=V.astype(theano.config.floatX))
        self.b = theano.shared(name='b', value=b.astype(theano.config.floatX))
        self.c = theano.shared(name='c', value=c.astype(theano.config.floatX))

        # Context-matrices + topic vector
        self.F = theano.shared(name='F', value=F.astype(theano.config.floatX))
        self.G = theano.shared(name='G', value=G.astype(theano.config.floatX))
        self.f = theano.shared(name='f', value=f.astype(theano.config.floatX))
        self.t = theano.shared(name='t', value=t.astype(theano.config.floatX))
        self.gamma = 0.2  # Gamma is used for exponential decay in context

        # SGD / rmsprop: Initialize parameters
        self.mE = theano.shared(name='mE', value=np.zeros(E.shape).astype(theano.config.floatX))
        self.mU = theano.shared(name='mU', value=np.zeros(U.shape).astype(theano.config.floatX))
        self.mV = theano.shared(name='mV', value=np.zeros(V.shape).astype(theano.config.floatX))
        self.mW = theano.shared(name='mW', value=np.zeros(W.shape).astype(theano.config.floatX))
        self.mb = theano.shared(name='mb', value=np.zeros(b.shape).astype(theano.config.floatX))
        self.mc = theano.shared(name='mc', value=np.zeros(c.shape).astype(theano.config.floatX))

        # Context matrices & vector
        self.mF = theano.shared(name='mF', value=np.zeros(F.shape).astype(theano.config.floatX))
        self.mG = theano.shared(name='mG', value=np.zeros(G.shape).astype(theano.config.floatX))
        self.mf = theano.shared(name='mf', value=np.zeros(f.shape).astype(theano.config.floatX))
        self.mt = theano.shared(name='mt', value=np.zeros(t.shape).astype(theano.config.floatX))

        # These are used to store Theano parameters
        self.theano = {}
        self.__theano_build__()
    
    def __theano_build__(self):
        E, V, U, W, b, c, F, G, f, t, gamma = self.E, self.V, self.U, self.W, self.b, self.c,\
                                    self.F, self.G, self.f, self.t, self.gamma
        
        x = T.ivector('x')
        y = T.ivector('y')

        def forward_prop_step(x_t, s_t1_prev, f_t1_prev):
            # Word embedding layer
            x_e = E[:,x_t]

            # Contextual computations
            f_t1 = 1 / np.random.normal() * (f_t1_prev ** gamma) * t ** (1 - gamma)  # Note, f & t are normalized,
            #  t gets updated for each word in beta matrix

            # LSTM Layer
            i_t1 = T.nnet.hard_sigmoid(U[0].dot(x_e) + W[0].dot(s_t1_prev) + b[0])
            j_t1 = T.nnet.hard_sigmoid(U[1].dot(x_e) + W[1].dot(s_t1_prev) + b[1])
            o_t1 = T.nnet.hard_sigmoid(U[2].dot(x_e) + W[2].dot(s_t1_prev) + b[2])
            g_t1 = T.tanh(U[3].dot(x_e) + W[3].dot(s_t1_prev) + b[3])
            s_t1 = (s_t1_prev * j_t1) + (g_t1 * i_t1) + (F[0].dot(f_t1))
            out = T.tanh(s_t1 * o_t1 + G[0].dot(f_t1))
            
            # Final output calculation
            # Theano's softmax returns a matrix with one row, we only need the row
            o_t = T.nnet.softmax(V.dot(out) + c)[0]
            return [o_t, s_t1, f_t1]
        
        [o, s, f], updates = theano.scan(
            forward_prop_step, sequences=x, truncate_gradient=self.bptt_truncate,
            outputs_info=[None,
                          dict(initial=T.zeros(self.hidden_dim)),
                          f])
        
        prediction = T.argmax(o, axis=1)
        o_error = T.sum(T.nnet.categorical_crossentropy(o, y))
        
        # Total cost (could add regularization here)
        cost = o_error
        
        # Gradients
        dE = T.grad(cost, E)
        dU = T.grad(cost, U)
        dW = T.grad(cost, W)
        db = T.grad(cost, b)
        dV = T.grad(cost, V)
        dc = T.grad(cost, c)

        # Assign functions
        self.forward_propagation = theano.function([x], o)
        self.predict_class = theano.function([x], prediction)
        self.ce_error = theano.function([x, y], cost)
        self.bptt = theano.function([x, y], [dE, dU, dW, db, dV, dc])
        
        # SGD parameters
        learning_rate = T.scalar('learning_rate')
        decay = T.scalar('decay')
        
        # rmsprop cache updates
        mE = decay * self.mE + (1 - decay) * dE ** 2
        mU = decay * self.mU + (1 - decay) * dU ** 2
        mW = decay * self.mW + (1 - decay) * dW ** 2
        mV = decay * self.mV + (1 - decay) * dV ** 2
        mb = decay * self.mb + (1 - decay) * db ** 2
        mc = decay * self.mc + (1 - decay) * dc ** 2
        
        self.sgd_step = theano.function(
            [x, y, learning_rate, theano.In(decay, value=0.9)],
            [], 
            updates=[(E, E - learning_rate * dE / T.sqrt(mE + 1e-6)),
                     (U, U - learning_rate * dU / T.sqrt(mU + 1e-6)),
                     (W, W - learning_rate * dW / T.sqrt(mW + 1e-6)),
                     (V, V - learning_rate * dV / T.sqrt(mV + 1e-6)),
                     (b, b - learning_rate * db / T.sqrt(mb + 1e-6)),
                     (c, c - learning_rate * dc / T.sqrt(mc + 1e-6)),
                     (self.mE, mE),
                     (self.mU, mU),
                     (self.mW, mW),
                     (self.mV, mV),
                     (self.mb, mb),
                     (self.mc, mc)
                    ])

    def calculate_total_loss(self, X, Y):
        return np.sum([self.ce_error(x,y) for x, y in zip(X,Y)])
    
    def calculate_loss(self, X, Y):
        # Divide calculate_loss by the number of words
        num_words = np.sum([len(y) for y in Y])
        return self.calculate_total_loss(X, Y) / float(num_words)


class LSTModel:
    def __init__(self, use_lda=False):

        # Read the data
        print "Reading data file..."

        self.file = smart_open.file_smart_open(FILE, 'rb')
        self.text = self.file.read().lower().decode('utf-8', errors='ignore')
        self.sentences = nltk.sent_tokenize(self.text)

        # Append SENTENCE_START and SENTENCE_END
        self.sentences = ["%s %s %s" % (sentence_start_token, x, sentence_end_token) for x in self.sentences]
        print "Parsed %d sentences." % (len(self.sentences))

        self.sentences = [re.sub('[.!?\r\n,:;]', '', sent) for sent in self.sentences]

        print "Tokenizing sentences..."
        # Tokenize the sentences into words
        self.tokenized_sentences = [nltk.word_tokenize(sent) for sent in self.sentences]

        # Count the word frequencies
        self.word_freq = nltk.FreqDist(itertools.chain(*self.tokenized_sentences))

        print "Found %d unique words tokens." % len(self.word_freq.items())

        # Get the most common words and build index_to_word and word_to_index vectors
        self.vocab = self.word_freq.most_common(vocabulary_size-1)
        self.index_to_word = [x[0] for x in self.vocab]
        self.index_to_word.append(unknown_token)
        self.word_to_index = dict([(w, i) for i, w in enumerate(self.index_to_word)])

        print "Using vocabulary size %d." % vocabulary_size
        print "The least frequent word in our vocabulary is '%s' and appeared %d times."\
              % (self.vocab[-1][0], self.vocab[-1][1])

        # Replace all words not in our vocabulary with the unknown token
        for i, sent in enumerate(self.tokenized_sentences):
            self.tokenized_sentences[i] = [w if w in self.word_to_index
                                           else unknown_token for w in sent]

        # Create the training data
        self.X_train = np.asarray([[self.word_to_index[w] for w in sent[:-1]]
                                   for sent in self.tokenized_sentences])
        self.y_train = np.asarray([[self.word_to_index[w] for w in sent[1:]]
                                   for sent in self.tokenized_sentences])

        # Use LDA
        if use_lda:
            print('Loading context information...')
            transformations = ['lda']
            # transformations = ['tfidf', 'lsi', 'lda', 'hdp']
            transformation_parameters = {'lda': {'load': True, 'topics': n_topics,
                                                 'cpu_cores': 16}}
            self.corpus = Corpus(override_corpus=False, override_dictionary=False,
                transf=transformations, transf_parameters=transformation_parameters)

            # Map context vector of given topic with dictionary:
            print "Creating topic matrix..."
            topic_wordid_matrix = np.zeros((n_topics, VOCABULARY_SIZE))
            topic_wordid_matrix.fill(0.000001)  # Introduce a small value to avoid zero probabilities

            for topic in range(0, n_topics):
                topic_vec = self.corpus.get_words_by_topic(topic)
                word_ids = list()
                for w_id in range(0, len(topic_vec)):
                    try:
                        word_ids.append((self.word_to_index.get(topic_vec[w_id][0]), topic_vec[w_id][1]))
                    except:
                        pass
                for wordid in range(0, len(word_ids)):
                    if wordid and word_ids[wordid][0]:
                        topic_wordid_matrix[int(topic), int(word_ids[wordid][0])] = word_ids[wordid][1]
            print "Topic matrix successfully created!"
            self.topic_wordid_matrix = topic_wordid_matrix

            # Load core model with LDA
            self.core_model = LSTMTheano(vocabulary_size, hidden_dim=HIDDEN_DIM,
                                         topic_matrix=self.topic_wordid_matrix)
        else:
             # Load core model w/o LDA
            self.core_model = LSTMTheano(vocabulary_size, hidden_dim=HIDDEN_DIM)

    def train_with_sgd(self, learning_rate=0.005, nepoch=1, evaluate_loss_after=5, epoch=0):
        # Keep track on losses
        losses = []
        num_examples_seen = 0
        print "Beginning network training..."
        for epoch in range(nepoch):
            print "Epoch #%s" % epoch
            # Optionally evaluate the loss
            if (epoch % evaluate_loss_after == 0):
                loss = self.core_model.calculate_loss(self.X_train, self.y_train)
                losses.append((num_examples_seen, loss))
                time = datetime.now().strftime('%Y-%m-%d-%H-%M-%S')

                print "%s: Loss after num_examples_seen=%d epoch=%d: %f" \
                      % (time, num_examples_seen, epoch, loss)
                # Adjust the learning rate if loss increases
                if (len(losses) > 1 and losses[-1][1] > losses[-2][1]):
                    learning_rate = learning_rate * 0.5
                    print "Setting learning rate to %f" % learning_rate
                sys.stdout.flush()

                # Saving model parameters
                save_model_parameters_theano("./trained_network/%s-lstm-epoch%s.npz"
                                             % (FILE.split("/")[-1].split(".")[-2], epoch),
                                             self.core_model)
            # For each training example...
            print "Training model..."
            for i in tqdm(range(len(self.y_train))):  # Also prints a progress bar
                # One SGD step
                self.core_model.sgd_step(self.X_train[i], self.y_train[i], learning_rate)
                num_examples_seen += 1

    def update_topic_word_vec(self, word_id):
        word_string = self.index_to_word[word_id]
        try:
            lda_word_id = self.corpus.dictionary.keys()[self.corpus.dictionary.values().index(word_string)]
            self.core_model.t = self.core_model.topic_matrix[:, lda_word_id-1]
            self.core_model.t = self.core_model.t / np.linalg.norm(self.core_model.t)  # Normalize t
        except ValueError:
            pass

    def generate_sentence(self):
        # Start the sentence with the start token
        new_sentence = [self.word_to_index[sentence_start_token]]
        # Repeat until we get an end token
        prev_sampled_word = self.word_to_index[unknown_token]
        while not new_sentence[-1] == self.word_to_index[sentence_end_token]:
            next_word_probs = self.core_model.forward_propagation(new_sentence)
            sampled_word = self.word_to_index[unknown_token]
            # We don't want to sample unknown words
            recursion = 0
            while (sampled_word == self.word_to_index[unknown_token] or sampled_word == prev_sampled_word)\
                    and recursion < 1000:
                samples = np.random.multinomial(1, next_word_probs[-1])
                sampled_word = np.argmax(samples)
                recursion += 1
            if sampled_word != self.word_to_index[sentence_start_token]:
                new_sentence.append(sampled_word)
                #self.update_topic_word_vec(sampled_word)  # update topic vec when appending a new word
            else:  # Sentence also ends if another start token is encountered
                break
            prev_sampled_word = sampled_word
            if len(new_sentence) > 100:  # Set a relevant recursion depth
                return u''
        sentence_str = [self.index_to_word[x] for x in new_sentence[1:-1]]
        return sentence_str