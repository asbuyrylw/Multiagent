
from sentence_transformers import SentenceTransformer
import faiss

class VectorMemory:
    def __init__(self):
        self.model = SentenceTransformer('all-MiniLM-L6-v2')
        self.index = faiss.IndexFlatL2(384)
        self.texts = []

    def add(self, text):
        vec = self.model.encode([text])
        self.index.add(vec)
        self.texts.append(text)

    def query(self, text, k=1):
        vec = self.model.encode([text])
        D, I = self.index.search(vec, k)
        return [self.texts[i] for i in I[0]]
