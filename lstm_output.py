#! /usr/bin/env python

# This file outputs text from trained network
import glob
import os
from lstm_model import load_model_parameters_theano, LSTModel, NEPOCH, LEARNING_RATE
import time
import codecs
from tqdm import tqdm


def main():
    lstm = LSTModel(use_lda=True)

    # Find model file, i.e. the file that is the newest:
    files = glob.glob('./trained_network/*.npz')
    latest_file = max(files, key=os.path.getctime)

    try:
        load_model_parameters_theano(latest_file, lstm.core_model)
    except IOError:
        print("Model file not found, training network instead!...")
        lstm.train_with_sgd(nepoch=NEPOCH, learning_rate=LEARNING_RATE)

    num_sentences = 100
    senten_min_length = 10

    print("Generating %d sentences with %d minimum length..." % (num_sentences, senten_min_length))

    all_sentences = []
    for i in tqdm(range(num_sentences)):
        sent = list()
        # We want long sentences, not sentences with one or two words
        while len(sent) < senten_min_length:
            sent = lstm.generate_sentence()
        print(" ".join(sent))
        all_sentences.append(" ".join(sent))

    outfile = codecs.open('./output/alice' + time.strftime("%Y%m%d-%H%M%S") + '.txt', 'w', 'utf-8')
    for item in all_sentences:
        outfile.write(item)
        outfile.write('. ')
    outfile.close()


if __name__ == '__main__':
    main()
