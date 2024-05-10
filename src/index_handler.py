import faiss

class IndexHandler:
    def __init__(self, norm):
        self.norm = norm
        self.index = faiss.IndexFlatL2() if norm == 'L2' else faiss.IndexFlatIP()
        self.idx_to_item = {}
        self.cur_idx = 0

    def add_items(self, sentences,embeddings):
        self.index.add(embeddings)
        for idx, sentence in enumerate(sentences):
            self.idx_to_item[self.cur_idx + idx] = sentence
        self.cur_idx += len(sentences)

    def __find_k_nearest_neighbors(self, query, k):
        distances, indices = self.index.search(query, k)
        return distances, indices

    def find_similar_sentences(self, query, k):
        query = query.reshape(1, -1)
        _, indices = self.__find_k_nearest_neighbors(query, k)
        similar_sentences = [self.idx_to_item[idx] for idx in indices[0]]
        return similar_sentences