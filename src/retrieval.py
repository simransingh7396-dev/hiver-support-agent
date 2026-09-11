"""
Retrieval over resolved_memory.jsonl
- TF-IDF baseline (no external deps) + optional sentence-transformers
- Used by agent.py to ground drafts in historical fixes
"""
import json
import re
from pathlib import Path
from typing import List, Dict

import sys
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

ROOT = Path(__file__).resolve().parent.parent
MEM_PATH = ROOT / "data" / "resolved_memory.jsonl"

class Retriever:
    def __init__(self, memory_path: Path = MEM_PATH, use_sbert: bool = False):
        self.memory_path = memory_path
        self.use_sbert = use_sbert
        self.entries: List[Dict] = []
        self.vectorizer = None
        self.tfidf_matrix = None
        self.sbert_model = None
        self.sbert_embeddings = None
        self._load()

    def _clean(self, text: str) -> str:
        text = re.sub(r"http\S+", "", text)
        text = re.sub(r"@\w+", "", text)
        return text.lower().strip()

    def _load(self):
        if not self.memory_path.exists():
            raise FileNotFoundError(f"Memory not found: {self.memory_path}. Run src/build_memory.py first")
        with open(self.memory_path, "r", encoding="utf-8") as f:
            for line in f:
                e = json.loads(line)
                # retrieval text = query + brand_resolution
                e["_retrieval_text"] = self._clean(e["query"] + " " + e["brand_resolution"])
                self.entries.append(e)
        print(f"Retriever loaded {len(self.entries)} entries from {self.memory_path}")

        # TF-IDF index
        corpus = [e["_retrieval_text"] for e in self.entries]
        self.vectorizer = TfidfVectorizer(max_features=8000, ngram_range=(1,2), stop_words="english", min_df=2)
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        print(f"TF-IDF matrix {self.tfidf_matrix.shape}")

        if self.use_sbert:
            try:
                from sentence_transformers import SentenceTransformer
                self.sbert_model = SentenceTransformer("all-MiniLM-L6-v2")
                texts = [e["query"] for e in self.entries]
                self.sbert_embeddings = self.sbert_model.encode(texts, show_progress_bar=True, normalize_embeddings=True)
                print(f"SBERT embeddings {self.sbert_embeddings.shape}")
            except Exception as e:
                print(f"SBERT load failed, falling back to TF-IDF: {e}")
                self.use_sbert = False

    def retrieve(self, query: str, k: int = 3) -> List[Dict]:
        q_clean = self._clean(query)
        if self.use_sbert and self.sbert_model is not None:
            q_emb = self.sbert_model.encode([query], normalize_embeddings=True)
            scores = (self.sbert_embeddings @ q_emb.T).flatten()
            idx = np.argsort(-scores)[:k]
            results = []
            for i in idx:
                e = self.entries[int(i)]
                results.append({**e, "score": float(scores[int(i)]), "method": "sbert"})
            return results
        else:
            q_vec = self.vectorizer.transform([q_clean])
            sims = cosine_similarity(q_vec, self.tfidf_matrix).flatten()
            idx = np.argsort(-sims)[:k]
            results = []
            for i in idx:
                e = self.entries[int(i)]
                results.append({**e, "score": float(sims[int(i)]), "method": "tfidf"})
            return results

    def retrieve_for_intent(self, query: str, intent: str = None, k: int = 3):
        # Simple: retrieve then you can filter downstream; for now just retrieve
        return self.retrieve(query, k=k)

if __name__ == "__main__":
    r = Retriever()
    tests = [
        "Error CE-34878-0 game keeps crashing",
        "I was charged twice for PS Plus refund please",
        "My controller won't sync to PS4"
    ]
    for q in tests:
        print("\nQuery:", q)
        for hit in r.retrieve(q, k=2):
            print(f"  [{hit['score']:.3f}] {hit['query'][:80]} -> {hit['brand_resolution'][:100]}")
