NOTE: this is an old repository with bad code

### Requirements
- Ubuntu 16.04 LTS, MacOS High Sierra or newer
- Python 3.6+


### How to setup
1. Setup virtualenvironment: `virtualenv venv -p python3`
2. Activate venv: `source venv/bin/activate`
3. Install requirements: `pip install -r requirements.txt`

### How to run LSTM + LDA

1. Calculate LDA beta matrix: `python calculate_lda.py`
2. Train network: `python train_lstm.py`
3. Generate output text: `python lstm_output.py`
4. Run analytics on generated text: `python analytics.py`

You can adjust n_topics and n_cpu_cores in `topic_modelling.py`.

Network parameters (vocabulary_size, hidden_dim, learning_rate, nepoch) can be adjusted in `lstm_model.py`

### Known issues

lstm_model gets stuck sometimes in the forward propagation step for an unknown reason
