import os
from src.embedder import Embedder
from src.index_handler import IndexHandler
from typing import Optional

class EmbeddingIndexer:
    '''
    This class is used to index text embeddings and retrieve similar text based on a query.

    Args:
    index_path (str): The path to save the index file. Default is None.
    norm (str): The norm to use for the index. Default is 'L2'.

    Attributes:
    index_path (str): The path to save the index file.
    embedder (Embedder): The embedder object used to encode text into embeddings.
    index_handler (IndexHandler): The index handler object used to handle the index.

    Methods:
    create_index_handler: Creates a new index handler object and saves it to the index path.
    load_index_handler: Loads an existing index handler object from the index path.
    read_sentences: Reads text sentences from a file.
    embed_sentences: Encodes text sentences into embeddings using the embedder object.
    '''
    def __init__(self, index_path: Optional[str], norm: str = 'L2'):
        self.index_path = index_path
        self.embedder = Embedder()
        self.index_handler = self.load_index_handler() if (index_path and os.path.exists(index_path)) else self.create_index_handler()

    def create_index_handler(self):
        sentences = self.read_sentences()
        embeddings = self.embedder.embed(sentences)
        index_handler = IndexHandler()
        index_handler.add_embeddings(sentences, embeddings)
        index_handler.save(self.index_path)
        return index_handler

    def load_index_handler(self):
        index_handler = IndexHandler()
        index_handler.load(self.index_path)
        return index_handler

    def read_sentences(self):
        sentences = []
        with open('data/meme_descriptions/descriptions.txt', 'r') as file:
            for line in file:
                sentences.append(line.strip())
        return sentences

    def embed_sentences(self, sentences: list[str]):
        embeddings = []
        for sentence in sentences:
            embedding = self.embedder.embed(sentence)
            embeddings.append(embedding)
        return embeddings