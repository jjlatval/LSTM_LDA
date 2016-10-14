#! /usr/bin/env python

# This file outputs text from trained network

from lstm_model import *
import time
import codecs

lstm = LSTModel(use_lda=True)

if MODEL_FILE != None:
    try:
        load_model_parameters_theano(MODEL_FILE, lstm.core_model)
        print "Model file found!"
    except IOError:
        print "Model file not found, training network instead!..."
        lstm.train_with_sgd(nepoch=NEPOCH, learning_rate=LEARNING_RATE)
else:
    print "Model file not found, training network instead!..."
    lstm.train_with_sgd(nepoch=NEPOCH, learning_rate=LEARNING_RATE)

num_sentences = 100
senten_min_length = 10

print "Generating %d sentences with %d minimum length..." % (num_sentences, senten_min_length)

all_sentences = []
for i in range(num_sentences):
    sent = list()
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = lstm.generate_sentence()
    print " ".join(sent)
    all_sentences.append(" ".join(sent))

outfile = codecs.open('./output/alice' + time.strftime("%Y%m%d-%H%M%S") +'.txt', 'w', 'utf-8')
for item in all_sentences:
    outfile.write(item)
    outfile.write('. ')
outfile.close()