from sklearn.model_selection import train_test_split
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub
from datetime import datetime

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization

from tensorflow import keras
import os
import re
import pandas as pd

def load_dataset(dir):
    file_name = "/users/TeamVideoSummarization/shivansh/bert/english/agr_en_train.csv"
    training = pd.read_csv(file_name, encoding='utf-8')
    training = training.fillna('nothing')
    arr = training.as_matrix()
    corpus = arr[:,1]
    y = arr[:,2]
    for i in range(len(y)):
        if (y[i] == 'NAG'):
            y[i] = 0
        elif (y[i] =='OAG'):
            y[i] = 2
        elif (y[i] == 'CAG'):
            y[i] = 1
    corpus = corpus.tolist()        
    y = y.astype(int)
    data = {}
    data["sentence"], data["polarity"] = corpus, y
    df = pd.DataFrame.from_dict(data)
    return df




# Download and process the dataset files.
def download_and_load_datasets(force_download=False):
  train_df = load_dataset("t")
  test_df = train_df
  return train_df, test_df



train, test = download_and_load_datasets()







train = train.sample(200) # TODO 
test = test.sample(200)

DATA_COLUMN = 'sentence'
LABEL_COLUMN = 'polarity'

label_list = [0, 1, 2]

train_InputExamples = train.apply(lambda x: bert.run_classifier.InputExample(guid=None, # Globally unique ID for bookkeeping, unused in this example
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)

test_InputExamples = test.apply(lambda x: bert.run_classifier.InputExample(guid=None, 
                                                                   text_a = x[DATA_COLUMN], 
                                                                   text_b = None, 
                                                                   label = x[LABEL_COLUMN]), axis = 1)



BERT_MODEL_HUB = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"

def create_tokenizer_from_hub_module():
  """Get the vocab file and casing info from the Hub module."""
  with tf.Graph().as_default():
    bert_module = hub.Module(BERT_MODEL_HUB)
    tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
    with tf.Session() as sess:
      vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                            tokenization_info["do_lower_case"]])
      
  return bert.tokenization.FullTokenizer(
      vocab_file=vocab_file, do_lower_case=do_lower_case)

tokenizer = create_tokenizer_from_hub_module()




# We'll set sequences to be at most 128 tokens long.
MAX_SEQ_LENGTH = 128
# Convert our train and test features to InputFeatures that BERT understands.
train_features = bert.run_classifier.convert_examples_to_features(train_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)
test_features = bert.run_classifier.convert_examples_to_features(test_InputExamples, label_list, MAX_SEQ_LENGTH, tokenizer)

del train_InputExamples
del test_InputExamples

def create_model(is_predicting, input_ids, input_mask, segment_ids, labels,
                 num_labels):
  """Creates a classification model."""
  bert_module = hub.Module(
      BERT_MODEL_HUB,
      trainable=True)
  bert_inputs = dict(
      input_ids=input_ids,
      input_mask=input_mask,
      segment_ids=segment_ids)
  bert_outputs = bert_module(
      inputs=bert_inputs,
      signature="tokens",
      as_dict=True)
  # Use "pooled_output" for classification tasks on an entire sentence.
  # Use "sequence_outputs" for token-level output.
  output_layer = bert_outputs["pooled_output"]
  hidden_size = output_layer.shape[-1].value
  # Create our own layer to tune for politeness data.
  output_weights = tf.get_variable(
      "output_weights", [num_labels, hidden_size],
      initializer=tf.truncated_normal_initial