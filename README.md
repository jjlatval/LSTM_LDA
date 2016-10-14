
#### Setup virtual environment

Clone the repository to your computer and create a virtual environment

$ git clone https://github.com/jjlatval/LSTM_LDA.git
$ cd LSTM_LDA
$ virtualenv venv
$ source venv/bin/activate

### Install requirements

In virtualenv run:
$ pip install -r requirements.txt

### How to run LSTM + LDA

1) Adjust n_topics and n_cpu_cores to your liking in topic_modelling.py
2) Calculate LDA beta matrix by running calculate_lda.py
3) Adjust network parameters (vocabulary_size, hidden_dim, learning_rate & nepoch) in lstm_model.py
4) Train network weights by running train_lstm.py
5) Generate output text by using the trained network in lstm_output.py
6) Run analytics on generated text by running analytics.py

### Known issues

lstm_model gets stuck sometimes in the forward propagation step for an unknown reason
