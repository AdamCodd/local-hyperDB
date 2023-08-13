import pytest
import numpy as np
from hyperdb import euclidean_metric, cosine_similarity, derridaean_similarity, adams_similarity, hyper_SVM_ranking_algorithm_sort, custom_ranking_algorithm_sort

def test_euclidean_metric_shape():
    data_vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
    query_vector = np.array([1, 1, 1])
    result = euclidean_metric(data_vectors, query_vector)
    assert result.shape == (3,), "Expected shape to be (3, ), but got different shape"

def test_cosine_similarity_values():
    data_vectors = np.array([[1, 0], [0, 1]])
    query_vector = np.array([1, 0])
    result = cosine_similarity(data_vectors, query_vector)
    assert np.array_equal(result, [1.0, 0.0]), "Expected [1.0, 0.0] but got different values"
    
def test_derridaean_similarity_shape():
    data_vectors = np.array([[1, 0], [0, 1]])
    query_vector = np.array([1, 0])
    result = derridaean_similarity(data_vectors, query_vector)
    assert result.shape == (2,), "Expected shape to be (2, ), but got different shape"

def test_adams_similarity_values():
    data_vectors = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    query_vector = np.array([1, 0])
    result = adams_similarity(data_vectors, query_vector)
    expected = [0.42, 0.42, 0.42]
    assert np.array_equal(result, expected), f"Expected {expected} but got different values"

def test_hyper_SVM_ranking_algorithm_sort():
    data_vectors = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    query_vector = np.array([1, 0])
    top_indices, similarities = hyper_SVM_ranking_algorithm_sort(data_vectors, query_vector)
    
    assert list(top_indices) == [0, 2, 1], "Indices are not ranked correctly"
    assert similarities[0] > similarities[1], "Similarity scores not sorted in descending order"

def test_custom_ranking_algorithm_sort_without_recency_bias():
    data_vectors = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    query_vector = np.array([1, 0])
    timestamps = ['1627825200.0', '1627911600.0', '1627998000.0']
    
    top_indices, combined_scores, similarities = custom_ranking_algorithm_sort(data_vectors, query_vector, timestamps, recency_bias=0)
    assert list(top_indices) == [0, 2, 1], "Indices are not ranked correctly without recency bias"
    
def test_custom_ranking_algorithm_sort_with_recency_bias():
    data_vectors = np.array([[1, 0], [0, 1], [0.5, 0.5]])
    query_vector = np.array([1, 0])
    timestamps = ['1627825200.0', '1627911600.0', '1627998000.0']
    
    top_indices, combined_scores, similarities = custom_ranking_algorithm_sort(data_vectors, query_vector, timestamps, recency_bias=1)
    assert list(top_indices) == [2, 0, 1], "Indices are not ranked correctly with recency bias"
    assert combined_scores[0] > combined_scores[1], "Combined scores not sorted in descending order"

# Test for edge cases
def test_euclidean_metric_empty_vectors():
    data_vectors = np.array([])
    query_vector = np.array([])
    with pytest.raises(ValueError):
        euclidean_metric(data_vectors, query_vector)
