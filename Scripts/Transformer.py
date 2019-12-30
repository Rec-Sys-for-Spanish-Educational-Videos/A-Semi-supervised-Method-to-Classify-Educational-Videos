import tensorflow_hub as hub
import tensorflow as tf
import numpy as np

def createNnlmEmbeddings(X):
   module = hub.load("https://tfhub.dev/google/nnlm-es-dim128-with-normalization/2")
   
   output = module(X)

   return output.numpy()