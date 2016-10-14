from utils import load_model_parameters_theano, save_model_parameters_theano
from rnn_theano import RNNTheano
import os
from train_theano import vocabulary_size, _HIDDEN_DIM, index_to_word, word_to_index,\
    sentence_end_token, sentence_start_token, unknown_token, model
import numpy as np

_VOCABULARY_SIZE = int(os.environ.get('VOCABULARY_SIZE', '3000'))  # orig 8000
_HIDDEN_DIM = int(os.environ.get('HIDDEN_DIM', '50'))  # orig 50
_LEARNING_RATE = float(os.environ.get('LEARNING_RATE', '0.005'))  # orig 0.005
_NEPOCH = int(os.environ.get('NEPOCH', '10'))  # orig 100
_MODEL_FILE = os.environ.get('MODEL_FILE')

print "a"
# losses = train_with_sgd(model, X_train, y_train, nepoch=50)
# save_model_parameters_theano('./data/trained-model-theano.npz', model)
print "Loading model parameters..."
load_model_parameters_theano('./data/rnn-theano-50-3000-2016-10-10-16-49-27.npz', model)


def generate_sentence(model):
    # We start the sentence with the start token
    new_sentence = [word_to_index[sentence_start_token]]
    # Repeat until we get an end token
    while not new_sentence[-1] == word_to_index[sentence_end_token]:
        next_word_probs = model.forward_propagation(new_sentence)
        sampled_word = word_to_index[unknown_token]
        # We don't want to sample unknown words
        while sampled_word == word_to_index[unknown_token]:
            samples = np.random.multinomial(1, next_word_probs[-1])
            sampled_word = np.argmax(samples)
        new_sentence.append(sampled_word)
    sentence_str = [index_to_word[x] for x in new_sentence[1:-1]]
    return sentence_str

num_sentences = 10
senten_min_length = 7

for i in range(num_sentences):
    sent = []
    # We want long sentences, not sentences with one or two words
    while len(sent) < senten_min_length:
        sent = generate_sentence(model)
    print " ".join(sent)