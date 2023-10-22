import numpy as np
import random
import warnings
from scipy.spatial import distance
from scipy.stats import pearsonr


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
    """
    Best for: Quantitative documents where the raw magnitude of terms is important.
    E.g., Scientific papers or technical reports.
    """
    similarities = np.dot(vectors, query_vector.T)
    return similarities

def cosine_similarity(vectors, query_vector):
    """
    Best for: General-purpose text documents where the orientation matters more than the magnitude.
    E.g., News articles, blog posts, or reviews.
    """
    norm_vectors = get_norm_vector(vectors)
    norm_query_vector = get_norm_vector(query_vector)

    # Compute cosine similarity
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities.flatten()
    
def euclidean_metric(vectors, query_vector, get_similarity_score=True):
    """
    Best for: Documents that are quantitatively comparable and are in a 'physical' space.
    E.g., Geospatial data or sensor readings.
    """
    similarities = np.linalg.norm(vectors - query_vector, axis=1)
    if get_similarity_score:
        similarities = 1 / (1 + similarities)
    return similarities

def manhattan_distance(vectors, query_vector):
    """
    Best for: Categorical or ordinal data where the sum of absolute differences is meaningful.
    E.g., Surveys or multiple-choice questionnaires.
    """
    distances = np.sum(np.abs(vectors - query_vector), axis=1)
    similarities = 1 / (1 + distances)  # Convert distance to similarity
    return similarities

def jaccard_similarity(vectors, query_vector):
    """
    Best for: Binary data or sets where only presence or absence of elements matters.
    E.g., Market basket analysis, recommendation systems.
    """
    vectors = check_and_binarize_vectors(vectors).astype(np.uint8)
    query_vector = check_and_binarize_vectors(query_vector).astype(np.uint8)
    # Calculate the intersection and union
    intersection = np.bitwise_and(vectors, query_vector)
    union = np.bitwise_or(vectors, query_vector)
    # Calculate Jaccard similarity
    jaccard_sim = np.sum(intersection, axis=1) / np.sum(union, axis=1)
    return jaccard_sim

def pearson_correlation(vectors, query_vector):
    """
    Best for: Documents where understanding the linear relationship between the variables is crucial.
    E.g., Time-series data, financial reports.
    Note: 
    - Returns NaN if both query_vector and a data_vector are constant.
    - Returns NaN if one of them is constant but not both.
    """

    query_vector = query_vector.flatten()

    # Calculate means and standard deviations upfront to avoid recomputation
    query_mean = np.mean(query_vector)
    vectors_mean = np.mean(vectors, axis=1)

    query_std = np.std(query_vector)
    vectors_std = np.std(vectors, axis=1)

    # Calculate the numerator (covariance)
    numerator = np.sum((vectors - vectors_mean[:, np.newaxis]) * (query_vector - query_mean), axis=1)

    # Calculate Pearson correlation
    denominator = vectors_std * query_std * vectors.shape[1]
    
    # Handle zero standard deviation cases
    mask = (denominator != 0)
    pearson_coeffs = np.zeros(vectors.shape[0])
    pearson_coeffs[mask] = numerator[mask] / denominator[mask]

    # Handle constant vectors
    constant_query = (query_std == 0)
    constant_vectors = (vectors_std == 0)

    pearson_coeffs[constant_query & constant_vectors] = np.nan  # Both are constant
    pearson_coeffs[constant_query ^ constant_vectors] = np.nan  # XOR (one is constant but not both)

    return pearson_coeffs


def check_and_binarize_vectors(vectors):
    # Check if vectors are binary
    unique_values = np.unique(vectors)
    if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
        return vectors  # Already binary
    
    # Binarize vectors in-place
    vectors[vectors > 0] = 1
    vectors[vectors <= 0] = 0
    
    return vectors

def hamming_distance(vectors, query_vector):
    """
    Best for: Binary data where small changes (flips) are crucial and each bit has equal importance.
    E.g., DNA sequences, error-correcting codes, or barcodes.
    """
    # Check and binarize vectors if necessary
    vectors = check_and_binarize_vectors(vectors).astype(np.uint8)
    query_vector = check_and_binarize_vectors(query_vector).astype(np.uint8)
    
    # XOR operation
    xor_result = np.bitwise_xor(vectors, query_vector)
    
    # Count set bits ('1's)
    hamming_dist = np.sum(np.unpackbits(xor_result, axis=1), axis=1)
    
    # Convert to similarity (lower distance means higher similarity)
    max_distance = vectors.shape[-1]  # Maximum possible Hamming distance
    similarities = max_distance - hamming_dist
    
    return similarities
    
def hyperDB_ranking_algorithm_sort(vectors, query_vector, top_k=5, metric='cosine_similarity', timestamps=None, recency_bias=0):    
    if np.isnan(vectors).any() or np.isnan(query_vector).any():
        raise ValueError("Vectors and query_vector should not contain NaN values.")

    vectors = np.array(vectors)
    # Calculate similarities using the provided metric
    metric_func = {
        'dot_product': dot_product,
        'cosine_similarity': cosine_similarity,
        'euclidean_metric': euclidean_metric,
        'manhattan_distance': manhattan_distance,
        'jaccard_similarity': jaccard_similarity,
        'pearson_correlation': pearson_correlation,
        'hamming_distance': hamming_distance
    }.get(metric, None)
    
    if metric_func is None:
        raise ValueError(f"Unknown metric: {metric}")
    
    similarities = metric_func(vectors, query_vector)
   
    # Ensure the similarities array is of type float
    similarities = similarities.astype(float)
    
    # Handle NaN values: replace NaNs with a very small number to ensure they are ranked last
    similarities[np.isnan(similarities)] = -np.inf
    
    #Debug
    #print("Similarities:", similarities)
    
    # If timestamps are provided, handle recency bias
    recency_scores = np.zeros(len(similarities))
    if timestamps is not None:
        if len(timestamps) > 0:
            recency_scores = recency_bias * np.exp(-np.max(timestamps) + timestamps)
            
    # Combine the similarities and the recency scores (if present)
    scores = similarities + recency_scores

    # Handle the case when there's only one document
    if np.array(scores).shape == () or (len(scores) == 1 and len(np.array(scores).shape) == 1):
        print("Info: Only one document left.")
        return np.array([0]), np.array([scores])

    # Efficiently fetch top-k indices
    if len(scores) > 0:
        actual_top_k = max(0, min(top_k, len(scores)))
        if actual_top_k > 0:
            # Flatten the scores array to make it 1D
            scores = scores.flatten()
            top_indices = np.argpartition(scores, -actual_top_k)[-actual_top_k:]
            top_indices = top_indices[np.argsort(-scores[top_indices])]
        else:
            return [], []

    return top_indices, scores[top_indices]
