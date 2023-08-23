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
    
def custom_ranking_algorithm_sort(vectors, query_vector, timestamps, top_k=5, metric=cosine_similarity, recency_bias=0):
    """HyperSVMRanking altered to take into account a recency_bias and favour more recent documents (recent memories)"""
    similarities = metric(vectors, query_vector)
    # Flatten the similarities array to 1-D if it's not already
    similarities = similarities.flatten()   
    # Convert string timestamps back to float
    float_timestamps = [float(timestamp) for timestamp in timestamps]
    if recency_bias > 0 and len(float_timestamps) > 0:
        max_timestamp = max(float_timestamps)
        # Compute the recency bias from the timestamps using an exponential decay function
        recency_scores = [recency_bias * np.exp(-(max_timestamp - timestamp)) for timestamp in float_timestamps]
    else:
        recency_scores = [0] * len(similarities)  # set recency_scores to 0 when not used
    # Combine the similarities and the recency scores
    combined_scores = [similarity + recency for similarity, recency in zip(similarities, recency_scores)]
    # Efficiently fetch top-k indices
    if len(combined_scores) > 0:
        actual_top_k = min(top_k, len(combined_scores))  # Limit top_k to the length of combined_scores
        top_indices = np.argpartition(combined_scores, -actual_top_k)[-actual_top_k:]
        # We still need to sort these top-k scores
        top_indices = top_indices[np.argsort(-np.array(combined_scores)[top_indices])]
    else:
        return [], [], []

    return top_indices, np.array(combined_scores)[top_indices], np.array(similarities)[top_indices]
