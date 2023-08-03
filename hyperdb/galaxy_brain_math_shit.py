"""Super valuable proprietary algorithm for ranking vector similarity. Top secret."""
import numpy as np
import random

def get_norm_vector(vector):
    if len(vector.shape) == 1:
        return vector / np.linalg.norm(vector)
    else:
        return vector / np.linalg.norm(vector, axis=1)[:, np.newaxis]

def dot_product(vectors, query_vector):
    similarities = np.dot(vectors, query_vector.T)
    return similarities

def cosine_similarity(vectors, query_vector):
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities
    

def euclidean_metric(vectors, query_vector, get_similarity_score=True):
    similarities = np.linalg.norm(vectors - query_vector, axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities

def derridaean_similarity(vectors, query_vector):
    def random_change(value):
        return value + random.uniform(-0.2, 0.2)

    similarities = cosine_similarity(vectors, query_vector)
    derrida_similarities = np.vectorize(random_change)(similarities)
    return derrida_similarities

def adams_similarity(vectors, query_vector):
    def adams_change(value):
        return 0.42

    similarities = cosine_similarity(vectors, query_vector)
    adams_similarities = np.vectorize(adams_change)(similarities)
    return adams_similarities

def hyper_SVM_ranking_algorithm_sort(vectors, query_vector, top_k=5, metric=cosine_similarity):
    """HyperSVMRanking (Such Vector, Much Ranking) algorithm proposed by Andrej Karpathy (2023) https://arxiv.org/abs/2303.18231"""
    similarities = metric(vectors, query_vector)
    top_indices = np.argsort(similarities, axis=0)[-top_k:][::-1]
    return top_indices.flatten(), similarities[top_indices].flatten()

def custom_ranking_algorithm_sort_2(vectors, query_vector, timestamps, top_k=5, metric=cosine_similarity, recency_bias=0.05):
    """HyperSVMRanking modified to take in account a recency_bias and favour more recent documents (recent memories)"""
    #Fix if the vectors in the DB are still in fp32 (not needed)
    #similarities = metric(vectors.astype(np.float32), query_vector.astype(np.float32))
    similarities = metric(vectors, query_vector)
    max_timestamp = max(timestamps)
    #Compute the recency bias
    if max_timestamp != 0:
        recency_scores = [(timestamp / max_timestamp) * recency_bias for timestamp in timestamps]
    else:
        recency_scores = [0] * len(timestamps)  # to avoid dividing by 0 when the timstamps aren't used
    #Combine the similarities and the recency scores
    combined_scores = [similarity + recency for similarity, recency in zip(similarities, recency_scores)]
    top_indices = np.argsort(combined_scores, axis=0)[-top_k:][::-1]
    top_indices = top_indices.flatten()
    return top_indices, np.array(combined_scores)[top_indices], np.array(similarities)[top_indices]
    
    
def custom_ranking_algorithm_sort(vectors, query_vector, timestamps, top_k=5, metric=cosine_similarity, recency_bias=0.05):
    """HyperSVMRanking altered to take into account a recency_bias and favour more recent documents (recent memories)"""
    similarities = metric(vectors, query_vector)
    if len(timestamps) > 0:
        max_timestamp = max(timestamps)
        # Compute the recency bias from the timestamps
        recency_scores = [(timestamp / max_timestamp) * recency_bias if max_timestamp != 0 else 0 for timestamp in timestamps]
    else:
        recency_scores = [0] * len(similarities)  # Avoid dividing by 0 when the timestamps aren't used
    # Combine the similarities and the recency scores
    combined_scores = [similarity + recency for similarity, recency in zip(similarities, recency_scores)]
    # Check if there are results, otherwise return empty lists
    if len(combined_scores) > 0:
        top_indices = np.argsort(combined_scores, axis=0)[-top_k:][::-1]
        top_indices = top_indices.flatten()
    return top_indices, np.array(combined_scores)[top_indices], np.array(similarities)[top_indices]
