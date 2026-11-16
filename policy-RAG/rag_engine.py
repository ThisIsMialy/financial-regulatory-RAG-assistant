import os
import re
from typing import Dict, List
import chromadb
from rank_bm25 import BM25Okapi
from google import genai
from google.genai import types


class HybridRAGEngine:
    def __init__(self, db_path: str = "./chroma_db", similarity_threshold: float = 0.30):
        self.similarity_threshold = similarity_threshold
        self.chroma = chromadb.PersistentClient(path=db_path)
        self.collection = self.chroma.get_collection(name="policy_docs")

        # Load corpus into memory for BM25 keyword search
        stored = self.collection.get(include=["documents", "metadatas"])
        self.ids: List[str] = stored["ids"]
        self.docs: List[str] = stored["documents"]
        self.metas: List[Dict] = stored["metadatas"]

        tokenized_corpus = [self._tokenize(doc) for doc in self.docs]
        self.bm25 = BM25Okapi(tokenized_corpus)

        self.llm = genai.Client(api_key=os.environ.get("GEMINI_API_KEY"))

    @staticmethod
    def _tokenize(text: str) -> List[str]:
        return re.findall(r"\w+", text.lower())

    def hybrid_retrieve(self, query: str, top_k: int = 4) -> Dict:
        # Dense Vector Search (Chroma cosine distance -> similarity = 1 - distance)
        vec_res = self.collection.query(
            query_texts=[query],
            n_results=min(top_k * 2, len(self.docs)),
            include=["documents", "metadatas", "distances"],
        )
        vec_ids = vec_res["ids"][0]
        vec_distances = vec_res["distances"][0]
        best_similarity = 1.0 - min(vec_distances) if vec_distances else 0.0

        # Guardrail: Refuse if semantic match is too weak
        if best_similarity < self.similarity_threshold:
            return {"passed_guardrail": False, "best_similarity": round(best_similarity, 3), "chunks": []}

        #  Sparse BM25 Keyword Search
        bm25_scores = self.bm25.get_scores(self._tokenize(query))
        bm25_ranked_indices = sorted(
            range(len(bm25_scores)), key=lambda i: bm25_scores[i], reverse=True
        )[: top_k * 2]
        bm25_ids = [self.ids[i] for i in bm25_ranked_indices if bm25_scores[i] > 0]

        # Reciprocal Rank Fusion (RRF)
        rrf_scores: Dict[str, float] = {}
        k_rrf = 60
        for rank, doc_id in enumerate(vec_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)
        for rank, doc_id in enumerate(bm25_ids):
            rrf_scores[doc_id] = rrf_scores.get(doc_id, 0.0) + 1.0 / (k_rrf + rank + 1)

        top_ids = sorted(rrf_scores, key=rrf_scores.get, reverse=True)[:top_k]
        id_to_idx = {doc_id: idx for idx, doc_id in enumerate(self.ids)}

        chunks = []
        for doc_id in top_ids:
            idx = id_to_idx[doc_id]
            chunks.append({
                "id": doc_id,
                "text": self.docs[idx],
                "source": self.metas[idx]["source"],
                "page": self.metas[idx]["page"],
            })

        return {
            "passed_guardrail": True,
            "best_similarity": round(best_similarity, 3),
            "chunks": chunks,
        }

    def answer(self, query: str, top_k: int = 4) -> Dict:
        retrieval = self.hybrid_retrieve(query, top_k=top_k)

        if not retrieval["passed_guardrail"]:
            return {
                "answer": "I cannot find verified information on this topic in the loaded policy documents.",
                "best_similarity": retrieval["best_similarity"],
                "sources": [],
            }

        context_blocks = []
        for c in retrieval["chunks"]:
            context_blocks.append(f"[{c['source']}, Page {c['page']}]: {c['text']}")
        context_str = "\n\n".join(context_blocks)

        system_instruction = (
            "You are a strict financial compliance assistant. Answer the user's question using ONLY "
            "the provided context excerpts. Every factual claim must end with an inline citation in "
            "the exact format [Filename, Page X]. If the context does not contain the answer, state "
            "that clearly and do not guess."
        )

        prompt = f"Context:\n{context_str}\n\nQuestion: {query}"

        response = self.llm.models.generate_content(
            model="gemini-2.5-flash",
            contents=prompt,
            config=types.GenerateContentConfig(
                system_instruction=system_instruction,
                temperature=0.1,
            ),
        )

        return {
            "answer": response.text,
            "best_similarity": retrieval["best_similarity"],
            "sources": retrieval["chunks"],
        }