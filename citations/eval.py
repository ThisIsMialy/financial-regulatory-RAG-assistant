from rag_engine import HybridRAGEngine

# Replace these sample questions & expected keywords/pages with 10 real checks from the PDFs
EVAL_SET = [
    {"query": "What are the customer due diligence requirements for e-KYC?", "expected_page": 5},
    {"query": "What is the minimum capital requirement?", "expected_page": 12},
]

## RUN after indexing PDFs
def run_eval():
    engine = HybridRAGEngine()
    hits = 0
    for item in EVAL_SET:
        res = engine.hybrid_retrieve(item["query"], top_k=4)
        retrieved_pages = [c["page"] for c in res["chunks"]]
        is_hit = item["expected_page"] in retrieved_pages
        hits += int(is_hit)
        print(f"Hit={is_hit} | Expected p.{item['expected_page']} | Got {retrieved_pages} | Q: {item['query']}")

    hit_rate = (hits / len(EVAL_SET)) * 100
    print(f"\nTop-4 Retrieval Hit Rate: {hit_rate:.1f}% ({hits}/{len(EVAL_SET)})")


if __name__ == "__main__":
    run_eval()