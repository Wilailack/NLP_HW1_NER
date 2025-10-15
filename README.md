# NLP_HW1_NER

NLP Assignment 1 - Named Entity Recognition

Description

Named entities are phrases that contain the names of persons, organizations, geo-
locations, companies, facilities etc.
e.g., U.N. official Ekeus (person) heads for Baghdad (geo-location)

The task of WNUT-2016 concerns named entity recognition. We will concentrate on
10 fine-grained NER categories: person, geo-location, company, facility, product,
music artist, movie, sports team, tv show and other. Dataset was extracted from
tweets in English language.

The named entity tags have format B-TYPE, I-TYPE, O. The first word of the entity
will have tag B-TYPE. I-TYPE means that the word is inside an entity of type
“TYPE”. Words with tag O do not belong to any entities.

For this assignment, you are offered training data, validation data and testing data (no
entity tags). You need to use the training data for training deep learning model, the
validation data for tuning hyper-parameters to improve performance of the model, the
testing data for final evaluation. When the training process is done, you use the well-
trained model to fill out the testing data with predicted named entity tags as
submission. We will evaluate your submissions with ground truth.

Requirements:

- Python programming language only.
- You can use any machine learning library ( TensorFlow, Keras, Pytorch, etc.).
- You can refer to the snippets of code from open-source projects, tutorials, blogs, etc. Do not clone the entire projects directly. Try to implement a model by yourself.
- Word embedding like Word2vec or Glove should be used in the project.

Grading:

We follow the definition of metrics introduced at CoNLL-2003 to measure the
performance of the systems in terms of precision, recall and F1-score, where:

“precision is the percentage of named entities found by the learning system that are
correct. Recall is the percentage of named entities present in the corpus that are
found by the system. A named entity is correct only if it is an exact match of the
corresponding entity in the data file.”

Models are evaluated based on exact-match F1-score on the testing data.

I choose to use the 840 billion tokens and 300 dimension vectors from glove embeddings: https://nlp.stanford.edu/projects/glove/

The conlleval.py is the evaluation function provided.

The evaluation from dev dataset score:
```
processed 16261 tokens with 661 phrases; found: 227 phrases; correct: 76.
accuracy:  11.88%; (non-O)
accuracy:  93.46%; precision:  33.48%; recall:  11.50%; FB1:  17.12
          company: precision:  56.25%; recall:  23.08%; FB1:  32.73  16
         facility: precision:  19.05%; recall:  10.53%; FB1:  13.56  21
          geo-loc: precision:  37.68%; recall:  22.41%; FB1:  28.11  69
            movie: precision:   0.00%; recall:   0.00%; FB1:   0.00  1
      musicartist: precision:   0.00%; recall:   0.00%; FB1:   0.00  4
            other: precision:  21.62%; recall:   6.06%; FB1:   9.47  37
           person: precision:  45.00%; recall:  15.79%; FB1:  23.38  60
          product: precision:   9.09%; recall:   2.70%; FB1:   4.17  11
       sportsteam: precision:  20.00%; recall:   1.43%; FB1:   2.67  5
           tvshow: precision:   0.00%; recall:   0.00%; FB1:   0.00  3
```