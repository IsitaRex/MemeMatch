from sentence_transformers import SentenceTransformer

class SentenceEmbedder:
  def __init__(self, model_name="sentence-transformers/all-MiniLM-L6-v2"):
    self.encoder = SentenceTransformer(model_name)

  def embed(self, sentences):
    embeddings = self.encoder.encode(sentences)
    return embeddings
