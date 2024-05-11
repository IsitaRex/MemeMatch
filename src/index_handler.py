import faiss
import numpy as np
from typing import Optional
class IndexHandler:
    '''
    This class is used to handle the indexing of text embeddings using faiss and retrieve similar text based on a query.

    Args:
    norm (str): The norm to use for the index. Default is 'L2'.
    dimension (int): The dimension of the embeddings. Default is 384.

    Attributes:
    norm (str): The norm to use for the index.
    index (faiss.IndexFlatL2 or faiss.IndexFlatIP): The faiss index object used to index text embeddings.
    idx_to_item (dict): A dictionary mapping index to text.
    cur_idx (int): The current index.

    Methods:
    add_embeddings: Adds text embeddings to the index.
    find_similar_sentences: Finds similar text based on a query.
    save: Saves the index to a file.
    load: Loads the index from a file.
    '''
    def __init__(self, norm: Optional[str] = 'L2', dimension: Optional[int] = 384):
        self.norm = norm
        self.index = faiss.IndexFlatL2(dimension) if norm == 'L2' else faiss.IndexFlatIP(dimension)
        self.idx_to_item = {}
        self.cur_idx = 0

    def add_embeddings(self, sentences: list[str],embeddings: list[np.array]):
        try :
            assert len(sentences) == len(embeddings)
        except AssertionError:
            raise ValueError("sentences and embeddings must have the same length")
        
        self.index.add(embeddings)
        for idx, sentence in enumerate(sentences):
            self.idx_to_item[self.cur_idx + idx] = sentence
        self.cur_idx += len(sentences)

    def __find_k_nearest_neighbors(self, query, k):
        distances, indices = self.index.search(query, k)
        return distances, indices

    def find_similar_sentences(self, query, k):
        _, indices = self.__find_k_nearest_neighbors(query, k)
        similar_sentences = [self.idx_to_item[idx] for idx in indices[0]]
        return similar_sentences, indices
    
    def save(self, path):
        faiss.write_index(self.index, path)
        with open(path.replace(".bin", ".txt"), "w") as f:
            for idx, item in self.idx_to_item.items():
                f.write(f"{idx}\t{item}\n")

    def load(self, path):
        self.index = faiss.read_index(path)
        with open(path.replace(".bin", ".txt"), "r") as f:
            for line in f:
                idx, item = line.strip().split("\t")
                self.idx_to_item[int(idx)] = item