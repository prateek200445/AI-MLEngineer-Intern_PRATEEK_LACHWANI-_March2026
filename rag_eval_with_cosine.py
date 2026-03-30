import json
import math
import os
from sentence_transformers import SentenceTransformer, util

# Path to your evaluation data (resolve relative to script to avoid backslash-escape issues)
eval_file_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "ragas_eval_data.json"))

# Check if file exists
if not os.path.exists(eval_file_path):
    raise FileNotFoundError(f"Evaluation file not found: {eval_file_path}")

# Load QA pairs
with open(eval_file_path, "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

print(f"Loaded {len(qa_pairs)} QA pairs for evaluation.")

# Load embedding model
model = SentenceTransformer("all-mpnet-base-v2")  


# Semantic similarity check
def is_relevant_semantic(expected, retrieved, threshold=0.6):
    emb_expected = model.encode(expected, convert_to_tensor=True)
    emb_retrieved = model.encode(retrieved, convert_to_tensor=True)
    sim = util.pytorch_cos_sim(emb_expected, emb_retrieved).item()
    return sim >= threshold, sim

# Compute embedding-based metrics

def compute_embedding_metrics(qa_pairs, k=5, threshold=0.6):
    
    total = sum(1 for item in qa_pairs if item.get("ground_truth") or item.get("reference") or item.get("answer"))
    hits = 0
    rr_sum = 0.0
    total_retrieved = 0
    total_relevant = 0
    ndcg_sum = 0.0
    avg_cosine_sum = 0.0

    for item in qa_pairs:
        
        expected = item.get("ground_truth") or item.get("reference") or item.get("answer") or ""
        
        retrieved_texts = (item.get("retrieved_texts") or item.get("contexts") or [])[:k]
        if not expected:
            
            continue

        found_rank = None
        relevance_scores = []

        for rank, txt in enumerate(retrieved_texts, start=1):
            relevant, sim = is_relevant_semantic(expected, txt, threshold)
            relevance_scores.append(1 if relevant else 0)
            avg_cosine_sum += sim
            if relevant and found_rank is None:
                found_rank = rank

        if found_rank is not None:
            hits += 1
            rr_sum += 1.0 / found_rank

        total_relevant += sum(relevance_scores)
        total_retrieved += len(retrieved_texts)

        
        # NDCG computation
        dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance_scores, start=1))
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, sum(relevance_scores) + 1))
        ndcg_sum += dcg / ideal_dcg if ideal_dcg > 0 else 0.0

    precision_at_k = total_relevant / total_retrieved if total_retrieved > 0 else 0
    recall_at_k = hits / total if total > 0 else 0
    mrr = rr_sum / total if total > 0 else 0
    ndcg_at_k = ndcg_sum / total if total > 0 else 0
    avg_cosine = avg_cosine_sum / (total_retrieved if total_retrieved > 0 else 1)

    return {
        "Precision@k": precision_at_k,
        "Recall@k": recall_at_k,
        "MRR": mrr,
        "NDCG@k": ndcg_at_k,
        "Avg_Cosine_Similarity": avg_cosine
    }

# Run metrics computation
metrics = compute_embedding_metrics(qa_pairs, k=5, threshold=0.6)

print("\nEmbedding-based Retrieval Evaluation Results:")
print(f"Precision@5: {metrics['Precision@k']:.3f}")
print(f"Recall@5: {metrics['Recall@k']:.3f}")
print(f"MRR: {metrics['MRR']:.3f}")
print(f"NDCG@5: {metrics['NDCG@k']:.3f}")
print(f"Avg Cosine Similarity: {metrics['Avg_Cosine_Similarity']:.3f}")
