import numpy as np
import random

def get_norm_vector(vector):
    norms = np.linalg.norm(vector, axis=-1, keepdims=True)
    zero_norm_indices = np.where(norms == 0)
    nan_indices = np.where(np.isnan(vector))

    if zero_norm_indices[0].size > 0:
        print(f"Warning: Vectors at indices {zero_norm_indices} have zero magnitude.")

    if nan_indices[0].size > 0:
        print(f"Warning: Vectors at indices {nan_indices} contain NaN values.")

    norm_vector = vector / norms
    return norm_vector

def dot_product(vectors, query_vector):
    similarities = np.dot(vectors, query_vector.T)
    return similarities

def cosine_similarity(vectors, query_vector):
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)
    # Compute cosine similarity
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities
    
def euclidean_metric(vectors, query_vector, get_similarity_score=True):
    similarities = np.linalg.norm(vectors - query_vector, axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities

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

    try:
        # Attempt to convert timestamps to floats
        float_timestamps = [float(timestamp) if timestamp is not None else 0.0 for timestamp in timestamps]
    except ValueError:
        print("Could not convert all timestamps to float. Defaulting to 0 for non-convertible timestamps.")
        float_timestamps = [float(timestamp) if isinstance(timestamp, (int, float)) else 0.0 for timestamp in timestamps]

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
