import tensorflow_hub as hub
import tensorflow as tf
from tensorflow import keras
from bert_serving.client import BertClient
import numpy as np
import bert
import os
import math

#Use model and tensorflow_text are available just on linux

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

class NnlmModel:
   def __init__(self):
      self.module = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")

   def createEmbeddings(self, X):
      return self.module(X).numpy()


class BertModel:
   def __init__(self):
      self.bc = BertClient()


   def createEmbeddings(self, X):
      return self.bc.encode(X)     
 