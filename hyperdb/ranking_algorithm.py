import numpy as np
import random
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
    norm_query_vector = get_norm_vector(np.atleast_2d(query_vector))

    # Compute cosine similarity
    similarities = np.dot(norm_vectors, norm_query_vector.T)
    return similarities
    
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
    query_vector = check_and_binarize_vectors(np.atleast_2d(query_vector)).astype(np.uint8)
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
    """
    # Initialize an array to hold the Pearson correlation coefficients
    query_vector = query_vector.flatten()
    pearson_coeffs = np.zeros(vectors.shape[0])
    for i, vec in enumerate(vectors):
        coeff, _ = pearsonr(vec, query_vector)
        pearson_coeffs[i] = coeff
    return pearson_coeffs

def mahalanobis_distance(vectors, query_vector):
    """
    Best for: Highly structured data with a known distribution and covariance.
    E.g., Medical data, scientific experiments.
    """
    # Calculate the mean and the inverse of the covariance matrix
    mean_vector = np.mean(vectors, axis=0)

    try:
        inv_cov_matrix = np.linalg.inv(np.cov(vectors, rowvar=False))
    except np.linalg.LinAlgError:
        print("Warning: Singular covariance matrix. Using pseudo-inverse instead.")
        inv_cov_matrix = np.linalg.pinv(np.cov(vectors, rowvar=False))

    # Calculate Mahalanobis distance
    delta_vector = query_vector - mean_vector
    distances = np.sqrt(np.dot(np.dot(delta_vector, inv_cov_matrix), delta_vector.T))

    # Convert to similarity (lower distance means higher similarity)
    similarities = 1 / (1 + distances)
    return similarities


def check_and_binarize_vectors(vectors):    
    # Check if vectors are binary
    unique_values = np.unique(vectors)
    if np.array_equal(unique_values, [0, 1]) or np.array_equal(unique_values, [0]) or np.array_equal(unique_values, [1]):
        return vectors  # Already binary
    
    # Binarize vectors
    binarized_vectors = np.where(vectors > 0, 1, 0)
    return binarized_vectors

def hamming_distance(vectors, query_vector):
    """
    Best for: Binary data where small changes (flips) are crucial and each bit has equal importance.
    E.g., DNA sequences, error-correcting codes, or barcodes.
    """
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
    if len(vectors.shape) != 2 and metric != 'cosine_similarity':
        vectors = vectors.reshape(vectors.shape[0], -1)
        
    if metric == 'hamming':
        vectors = vectors.reshape(vectors.shape[0], -1)
        similarities = hamming_distance(vectors, query_vector)
    elif metric == 'dot_product':
        similarities = dot_product(vectors, query_vector)
    elif metric == 'cosine_similarity':
        similarities = cosine_similarity(vectors, query_vector)
    elif metric == 'euclidean_metric':
        similarities = euclidean_metric(vectors, query_vector)
    elif metric == 'manhattan_distance':
        similarities = manhattan_distance(vectors, query_vector)
    elif metric == 'jaccard_similarity':
        similarities = jaccard_similarity(vectors, query_vector)
    elif metric == 'pearson_correlation':
        similarities = pearson_correlation(vectors, query_vector)
    elif metric == 'mahalanobis_distance':
        similarities = np.array([mahalanobis_distance(vectors, query_vector)] * vectors.shape[0])
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
