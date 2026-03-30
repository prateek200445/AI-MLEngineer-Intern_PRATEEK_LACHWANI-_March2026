import json
from qdrant_client import QdrantClient
from langchain_community.vectorstores import Qdrant
from langchain_community.embeddings import HuggingFaceBgeEmbeddings
from fuzzywuzzy import fuzz
from config import settings

#step1 for loading the dataset
def load_qa_from_json(path):
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)

    qa_pairs = []
    for item in data:
        query = item.get("query")
        answer = item.get("ground_truth")
        if not query or not answer:
            raise ValueError(f"Invalid item: {item}")
        qa_pairs.append({"question": query, "ground_truth": answer})
    return qa_pairs


qa_pairs = load_qa_from_json("cloud_questions_answers.json")
print(f"Loaded {len(qa_pairs)} question–answer pairs")
print(qa_pairs[:2])


#step2 connect the qdrant
embeddings = HuggingFaceBgeEmbeddings(
    model_name=settings.EMBEDDING_MODEL,
    model_kwargs={"device": settings.EMBEDDING_DEVICE}
)

if settings.QDRANT_URL:
    qdrant_url = settings.QDRANT_URL
else:
    qdrant_url = f"http://{settings.QDRANT_HOST}:{settings.QDRANT_PORT}"

client = QdrantClient(
    url=qdrant_url,
    prefer_grpc=settings.PREFER_GRPC,
    api_key=settings.QDRANT_API_KEY
)

db = Qdrant(
    collection_name=settings.QDRANT_COLLECTION,
    embeddings=embeddings,
    client=client
)

print(f" Connected to Qdrant collection: {settings.QDRANT_COLLECTION}")



k = 5  # number of top results per question
print(f" Retrieving top {k} contexts for each question...")

for idx, item in enumerate(qa_pairs, 1):
    q = item["question"]
    results = db.similarity_search_with_score(q, k=k)

    item["retrieved_texts"] = [doc.page_content for doc, _ in results]
    item["retrieved_scores"] = [score for _doc, score in results]

    print(f"{idx}. Retrieved {len(results)} chunks for: {q[:60]}...")

#saving retrival results
with open("retrieval_results.json", "w", encoding="utf-8") as f:
    json.dump(qa_pairs, f, ensure_ascii=False, indent=2)

print(" Retrieval results saved to retrieval_results.json")



with open("retrieval_results.json", "r", encoding="utf-8") as f:
    qa_pairs = json.load(f)

print(f" Loaded {len(qa_pairs)} retrieval results for evaluation.")


from fuzzywuzzy import fuzz
import math

def is_relevant(expected, text, threshold=50):
    return fuzz.partial_ratio(expected.lower(), text.lower()) >= threshold

def compute_metrics(qa_pairs, k=5):
    total = len(qa_pairs)
    hits = 0
    rr_sum = 0.0
    total_retrieved = 0
    total_relevant = 0
    ndcg_sum = 0.0

    for item in qa_pairs:
        expected = item["ground_truth"]
        retrieved_texts = item.get("retrieved_texts", [])[:k]
        found_rank = None
        relevance_scores = []

        for rank, txt in enumerate(retrieved_texts, start=1):
            relevant = 1 if is_relevant(expected, txt) else 0
            relevance_scores.append(relevant)
            if relevant and found_rank is None:
                found_rank = rank

        if found_rank is not None:
            hits += 1
            rr_sum += 1.0 / found_rank

        total_relevant += sum(relevance_scores)
        total_retrieved += len(retrieved_texts)

        dcg = sum(rel / math.log2(rank + 1) for rank, rel in enumerate(relevance_scores, start=1))
        ideal_dcg = sum(1.0 / math.log2(rank + 1) for rank in range(1, sum(relevance_scores) + 1))
        ndcg = dcg / ideal_dcg if ideal_dcg > 0 else 0.0
        ndcg_sum += ndcg

    recall_at_k = hits / total if total > 0 else 0
    mrr = rr_sum / total if total > 0 else 0
    precision_at_k = total_relevant / total_retrieved if total_retrieved > 0 else 0
    ndcg_at_k = ndcg_sum / total if total > 0 else 0

    return {
        "Recall@k": recall_at_k,
        "Precision@k": precision_at_k,
        "MRR": mrr,
        "NDCG@k": ndcg_at_k
    }



metrics = compute_metrics(qa_pairs, k=5)

print("\n Retrieval Evaluation Results:")
print(f"Recall@5: {metrics['Recall@k']:.3f}")
print(f"Precision@5: {metrics['Precision@k']:.3f}")
print(f"MRR: {metrics['MRR']:.3f}")
print(f"NDCG@5: {metrics['NDCG@k']:.3f}")
