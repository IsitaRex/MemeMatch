import numpy as np
from sentence_transformers import SentenceTransformer

class Embedder:
  '''
  This class is used to encode text into embeddings using a pre-trained sentence transformer model.

  Args:
  model_name (str): The name of the pre-trained sentence transformer model to use. Default is "sentence-transformers/all-MiniLM-L6-v2".

  Attributes:
  encoder (SentenceTransformer): The sentence transformer model used to encode text into embeddings.

  Methods:
  embed: Encodes text into embeddings using the sentence transformer model.
  '''

  def __init__(self, model_name: str ="sentence-transformers/all-MiniLM-L6-v2"):
    self.encoder = SentenceTransformer(model_name)

  def embed(self, sentences: list[str]):
    embeddings = self.encoder.encode(sentences)
    return np.stack(embeddings)
