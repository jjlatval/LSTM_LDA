#! /usr/bin/env python

# This file can be used to train the LSTM model

from lstm_model import *

train_loaded = False

lstm = LSTModel()

print "Training network..."

if train_loaded:
    try:
        load_model_parameters_theano(MODEL_FILE, lstm.core_model)
        epoch = int(MODEL_FILE.split(".")[-2].split("epoch")[-1])
        print "Model file found! Continuing to train this model from epoch %d" % epoch
        lstm.train_with_sgd(nepoch=NEPOCH, learning_rate=LEARNING_RATE, epoch=epoch)
    except IOError:
        print "Model file not found, training network instead!..."
        lstm.train_with_sgd(nepoch=NEPOCH, learning_rate=LEARNING_RATE)
else:
    lstm.train_with_sgd(nepoch=NEPOCH, learning_rate=LEARNING_RATE)
