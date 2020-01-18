import tensorflow_hub as hub
import tensorflow as tf

from tensorflow import keras
import numpy as np
import bert
import os
import math

#Use model and tensorflow_text are available just on linux
"""
import tensorflow_text
class UseModel:
   def __init__(self):
      self.module = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual/3")
      self.batch_size = 100
         
   def createEmbeddings(self, X):
      embeddings = []
      for i in range(0, len(X) // self.batch_size):
              l = i*self.batch_size 
              r = l + self.batch_size
              embeddings.append(self.module(X[l:r]).numpy())
      
      if(len(X) % self.batch_size != 0):
          l = (len(X) // self.batch_size) * self.batch_size 
          r = len(X)
          embeddings.append(self.module(X[l:r]).numpy())

      list_2d = []
      for batch in embeddings:
         for embedding in batch:
            list_2d.append(embedding)

      return np.array(list_2d)
"""
class NnlmModel:
   def __init__(self):
      self.module = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")

   def createEmbeddings(self, X):
      return self.module(X).numpy()


class BertModel:
   def __init__(self):
      self.model_dir = "PretrainedModels/multilingual_L-12_H-768_A-12"
      self.max_seq_len = 64

      bert_params = bert.params_from_pretrained_ckpt(self.model_dir)
      l_bert = bert.BertModelLayer.from_params(bert_params, name = "bert")
      l_input_ids = keras.layers.Input(shape=(self.max_seq_len,), dtype='int32')
      
      model = keras.Sequential()
      
      model.add(l_input_ids)
      model.add(l_bert)
      model.add(keras.layers.GlobalAveragePooling1D())

      output = model(l_input_ids)

      self.model = keras.Model(inputs=l_input_ids, outputs=output)
      
      self.model.build(input_shape=(None, self.max_seq_len))

      bert_ckpt_file   = os.path.join(self.model_dir, "bert_model.ckpt")
      bert.load_bert_weights(l_bert, bert_ckpt_file)


   def createEmbeddings(self, X):

      vocab_file = os.path.join(self.model_dir, 'vocab.txt') 
      tokenizer = bert.bert_tokenization.FullTokenizer(vocab_file, True)

      embeddings_list = []
      embeddedSentences = 0
      token_ids_list = []
      nr=0
      for sentence in X:
         nr+=1
         tokens = tokenizer.tokenize(sentence)
         for window_idx in range(0, len(tokens) // self.max_seq_len + 1):
            start = window_idx * self.max_seq_len
            end = start + self.max_seq_len

            chopped_window = tokens[start:min(end, len(tokens))]

            for remain in range(len(chopped_window), self.max_seq_len):
               chopped_window.append("[PAD]")

            token_ids = tokenizer.convert_tokens_to_ids(chopped_window)

            token_ids_list.append(token_ids)

      token_ids_batch = np.array(token_ids_list).reshape((len(token_ids_list), self.max_seq_len))

      windows_prediction = self.model.predict(token_ids_batch)

      window_global_idx = 0;
      for sentence in X:
         tokens = tokenizer.tokenize(sentence)
         windows_embedding_list = []

         sentence_embedding = np.zeros(768)

         for window_idx in range(0, len(tokens) // self.max_seq_len + 1):
            sentence_embedding+=windows_prediction[window_global_idx + window_idx]
         
         embeddings_list.append(sentence_embedding)
         window_global_idx += len(tokens) // self.max_seq_len + 1;


      return np.array(embeddings_list)
 