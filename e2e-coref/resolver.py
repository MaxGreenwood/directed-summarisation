from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import input
import tensorflow.compat.v1 as tf
import coref_model as cm
import util
tf.disable_v2_behavior()
from ast import literal_eval
import pandas as pd

import nltk
nltk.download("punkt")
from nltk.tokenize import sent_tokenize, word_tokenize

def create_example(text):
  raw_sentences = sent_tokenize(text)
  sentences = [word_tokenize(s) for s in raw_sentences]
  speakers = [["" for _ in sentence] for sentence in sentences]
  return {
    "doc_key": "nw",
    "clusters": [],
    "sentences": sentences,
    "speakers": speakers,
  }

def print_predictions(example):
  words = util.flatten(example["sentences"])
  for cluster in example["predicted_clusters"]:
    print(u"Predicted cluster: {}".format([" ".join(words[m[0]:m[1]+1]) for m in cluster]))

def make_predictions(text, model):
  example = create_example(text)
  tensorized_example = model.tensorize_example(example, is_training=False)
  feed_dict = {i:t for i,t in zip(model.input_tensors, tensorized_example)}
  _, _, _, mention_starts, mention_ends, antecedents, antecedent_scores, head_scores = session.run(model.predictions + [model.head_scores], feed_dict=feed_dict)

  predicted_antecedents = model.get_predicted_antecedents(antecedents, antecedent_scores)

  example["predicted_clusters"], _ = model.get_predicted_clusters(mention_starts, mention_ends, predicted_antecedents)
  example["top_spans"] = zip((int(i) for i in mention_starts), (int(i) for i in mention_ends))
  example["head_scores"] = head_scores.tolist()
  return example

if __name__ == "__main__":
  config = util.initialize_from_env()
  model = cm.CorefModel(config)
  with tf.Session() as session:
    model.restore(session)

    # Read in dataframe and create empty column
    corpus = pd.read_csv('coref_input.csv', converters={"sentences": literal_eval})
    predicted_clusters = []
    top_spans = []
    head_scores = []

    # Create story for input
    for idx, art in corpus.iterrows():
      text = " ".join(art['sentences']) # list -> string

      # Make predictions and append outputs to lists
      if len(text) > 0:
        example = make_predictions(text, model)
        predicted_clusters.append(example['predicted_clusters'])
        top_spans.append(example['top_spans'])
        head_scores.append(example['head_scores'])
        print_predictions(make_predictions(text, model))

  # Update dataframe and export
  corpus['predicted_clusters'] = predicted_clusters
  corpus['top_spans'] = top_spans
  corpus['head_scores'] = head_scores
  corpus.to_csv('coref.csv')
