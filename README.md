# NLP_HW1_NER

NLP homework no.1 - Name Entity Recognition

I should to use the 840 billion tokens and 300 dimension vectors from glove embeddings: https://nlp.stanford.edu/projects/glove/

The conlleval.py is the evaluation function provided.

The evaluation from dev dataset score:
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