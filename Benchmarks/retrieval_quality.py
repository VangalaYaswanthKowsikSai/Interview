import numpy as np
from sklearn.metrics.pairwise import cosine_similarity

def calculate_metrics(model_wrapper, corpus, queries, ground_truth):
    """
    Calculates Recall@K and NDCG@K.
    """
    print(f"  ...Encoding corpus ({len(corpus)} docs)...")
    corpus_embeddings = model_wrapper.encode(corpus)
    
    print(f"  ...Encoding queries ({len(queries)})...")
    query_embeddings = model_wrapper.encode(queries)

    # Calculate Similarity Matrix
    similarities = cosine_similarity(query_embeddings, corpus_embeddings)

    recall_1 = []
    recall_5 = []
    ndcg_10 = []

    for i, query_sims in enumerate(similarities):
        # Get indices of top matches
        sorted_indices = np.argsort(query_sims)[::-1]
        
        relevant_doc_index = ground_truth[i] # We assume 1 relevant doc per query for this test

        # Recall @ 1
        recall_1.append(1 if relevant_doc_index in sorted_indices[:1] else 0)
        
        # Recall @ 5
        recall_5.append(1 if relevant_doc_index in sorted_indices[:5] else 0)

        # NDCG @ 10 (Simplified for binary relevance)
        if relevant_doc_index in sorted_indices[:10]:
            rank = list(sorted_indices).index(relevant_doc_index) + 1
            ndcg_10.append(1 / np.log2(rank + 1))
        else:
            ndcg_10.append(0.0)

    return {
        "recall@1": np.mean(recall_1),
        "recall@5": np.mean(recall_5),
        "ndcg@10": np.mean(ndcg_10)
    }