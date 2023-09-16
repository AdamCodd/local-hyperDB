import numpy as np
import random

def get_norm_vector(vector):
    norms = np.linalg.norm(vector, axis=-1, keepdims=True)
    
    zero_norm_indices = np.where(norms == 0)
    nan_indices = np.where(np.isnan(vector))

    if zero_norm_indices[0].size > 0:
        norms[norms == 0] = 1  # Avoid division by zero by setting the norm to 1

    if nan_indices[0].size > 0:
        print(f"Warning: Vectors at indices {nan_indices} contain NaN values.")
        
    norm_vector = vector / norms
    return norm_vector


def dot_product(vectors, query_vector):
    similarities = np.dot(vectors, query_vector.T)
    return similarities

def cosine_similarity(vectors, query_vector):
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(np.atleast_2d(query_vector))

    # Compute cosine similarity
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities
    
def euclidean_metric(vectors, query_vector, get_similarity_score=True):
    similarities = np.linalg.norm(vectors - query_vector, axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities

def check_and_binarize_vectors(vectors):
    print("Unique values before binarization:", np.unique(vectors))
    
    # Check if vectors are binary
    unique_values = np.unique(vectors)
    if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
        return vectors  # Already binary
    
    # Binarize vectors
    binarized_vectors = np.where(vectors > 0, 1, 0)
    
    print("Unique values after binarization:", np.unique(binarized_vectors))
    return binarized_vectors

def hamming_distance(vectors, query_vector):
    # Check and binarize vectors if necessary
    vectors = check_and_binarize_vectors(vectors).astype(np.uint8)
    query_vector = check_and_binarize_vectors(np.atleast_2d(query_vector)).astype(np.uint8)
    
    # XOR operation
    xor_result = np.bitwise_xor(vectors, query_vector)
    
    # Count set bits ('1's)
    hamming_dist = np.sum(np.unpackbits(xor_result, axis=1), axis=1)
    
    # Convert to similarity (lower distance means higher similarity)
    max_distance = vectors.shape[-1]  # Maximum possible Hamming distance
    similarities = max_distance - hamming_dist
    
    return similarities

    
def hyperDB_ranking_algorithm_sort(vectors, query_vector, top_k=5, metric='cosine_similarity', timestamps=None, recency_bias=0):
    # Calculate similarities using the provided metric
    if metric == 'hamming':
        vectors = vectors.reshape(vectors.shape[0], -1)
        similarities = hamming_distance(vectors, query_vector)
    elif metric == 'dot_product':
        similarities = dot_product(vectors, query_vector)
    elif metric == 'cosine_similarity':
        similarities = cosine_similarity(vectors, query_vector)
    elif metric == 'euclidean_metric':
        vectors = vectors.reshape(vectors.shape[0], -1)
        similarities = euclidean_metric(vectors, query_vector)
    else:
        raise ValueError(f"Unknown metric: {metric}")
    
    # Flatten the similarities array to 1-D if it's not already
    similarities = similarities.flatten()
    
    # If timestamps are provided, handle recency bias
    if timestamps is not None:
        try:
            float_timestamps = [float(timestamp) if timestamp is not None else 0.0 for timestamp in timestamps]
        except ValueError:
            print("Could not convert all timestamps to float. Defaulting to 0 for non-convertible timestamps.")
            float_timestamps = [float(timestamp) if isinstance(timestamp, (int, float)) else 0.0 for timestamp in timestamps]
        
        if recency_bias > 0 and len(float_timestamps) > 0:
            max_timestamp = max(float_timestamps)
            recency_scores = [recency_bias * np.exp(-(max_timestamp - timestamp)) for timestamp in float_timestamps]
        else:
            recency_scores = [0] * len(similarities)
        
        # Combine the similarities and the recency scores
        combined_scores = [similarity + recency for similarity, recency in zip(similarities, recency_scores)]
    else:
        combined_scores = similarities  # If no timestamps, then combined_scores = original similarities
        
    # Handle the case when there's only one document
    if np.array(similarities).shape == ():
        print("Info: Only one document left.")
        return np.array([0]), np.array([similarities]), np.array([similarities])
    
    # Efficiently fetch top-k indices
    if len(combined_scores) > 0:
        actual_top_k = min(top_k, len(combined_scores))
        top_indices = np.argpartition(combined_scores, -actual_top_k)[-actual_top_k:]
        top_indices = top_indices[np.argsort(-np.array(combined_scores)[top_indices])]
    else:
        return [], [], []
        
    return top_indices, np.array(combined_scores)[top_indices], np.array(similarities)[top_indices]
