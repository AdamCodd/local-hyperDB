import pytest
import numpy as np
from hyperdb.ranking_algorithm import (euclidean_metric, cosine_similarity, manhattan_distance, 
                         jaccard_similarity, pearson_correlation, hamming_distance, hyperDB_ranking_algorithm_sort)

class TestEuclideanMetric:
    def test_shape_and_values(self):
        """Test if the shape and values of the output array are correct"""
        # Combined test for shape and some basic value checking
        data_vectors = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])
        query_vector = np.array([1, 1, 1])
        result = euclidean_metric(data_vectors, query_vector)
        assert result.shape == (3,), "Expected shape to be (3, ), but got different shape"
        assert np.all(result > 0), "All values should be positive"

    def test_empty_vectors(self):
        """Test the function with empty vectors"""
        data_vectors = np.array([])
        query_vector = np.array([])
        with pytest.raises(ValueError):
            euclidean_metric(data_vectors, query_vector)

class TestCosineSimilarity:
    def test_values(self):
        """Test if the values in the output array are correct"""
        data_vectors = np.array([[1, 0], [0, 1]])
        query_vector = np.array([1, 0])
        result = cosine_similarity(data_vectors, query_vector)
        assert np.array_equal(result, [1.0, 0.0]), "Expected [1.0, 0.0] but got different values"

class TestManhattanDistance:
    def test_basic_case(self):
        """Test basic functionality of Manhattan distance"""
        data_vectors = np.array([[1, 0], [0, 1]])
        query_vector = np.array([1, 0])
        result = manhattan_distance(data_vectors, query_vector)
        assert np.allclose(result, [1.0, 1/3]), f"Expected [1.0, 0.3333] but got {result}"

class TestJaccardSimilarity:
    def test_basic_case(self):
        """Test basic functionality of Jaccard similarity"""
        data_vectors = np.array([[1, 1], [1, 0], [0, 0]])
        query_vector = np.array([1, 1])
        result = jaccard_similarity(data_vectors, query_vector)
        assert np.array_equal(result, [1.0, 0.5, 0.0])

    def test_non_binary_vectors(self):
        """Test Jaccard similarity with non-binary vectors"""
        data_vectors = np.array([[2, 2], [2, 0], [0, 0]])
        query_vector = np.array([1, 1])
        result = jaccard_similarity(data_vectors, query_vector)
        assert np.array_equal(result, [1.0, 0.5, 0.0])

class TestPearsonCorrelation:
    def test_basic_case(self):
        """Test basic functionality of Pearson correlation"""
        data_vectors = np.array([[1, 1], [0, 1], [1, 0]])
        query_vector = np.array([1, 1])
        result = pearson_correlation(data_vectors, query_vector)
        assert np.isnan(result[0])  # Both query and first data vector are constant
        assert result[1] != 0.0  # Not constant vectors
        assert result[2] != 0.0  # Not constant vectors

    def test_constant_vectors(self):
        """Test behavior with constant vectors"""
        data_vectors = np.array([[1, 1], [0, 0], [1, 1]])
        query_vector = np.array([1, 1])
        result = pearson_correlation(data_vectors, query_vector)
        assert np.isnan(result[0])  # Both query and first data vector are constant
        assert np.isnan(result[1])  # Both query and second data vector are constant
        assert np.isnan(result[2])  # Both query and third data vector are constant

class TestHammingDistance:
    def test_basic_case(self):
        """Test basic functionality of Hamming distance"""
        data_vectors = np.array([[1, 1], [0, 1], [1, 0]])
        query_vector = np.array([1, 1])
        result = hamming_distance(data_vectors, query_vector)
        assert np.array_equal(result, [2, 1, 1])

class TestHyperDBRankingAlgorithmSort:
    @pytest.mark.parametrize("metric, recency_bias, expected_indices",
                             [("cosine_similarity", 0, [0, 2, 1]),
                              ("cosine_similarity", 1, [2, 0, 1]),
                              ("euclidean_metric", 0, [0, 2, 1]),
                              ("manhattan_distance", 0, [0, 2, 1]),
                              ("jaccard_similarity", 0, [0, 2, 1]),
                              ("pearson_correlation", 0, [0, 1, 2]),
                              ("hamming_distance", 0, [0, 2, 1])
                              ])
    def test_custom_ranking_algorithm_sort(self, metric, recency_bias, expected_indices):
        """Test if the function returns correct top indices with and without recency bias"""
        data_vectors = np.array([[1, 0], [0, 1], [0.5, 0.5]])
        query_vector = np.array([1, 0])
        timestamps = [1627825200.0, 1627911600.0, 1627998000.0]
        
        top_indices, _ = hyperDB_ranking_algorithm_sort(data_vectors, query_vector, metric=metric, timestamps=timestamps, recency_bias=recency_bias)
        assert list(top_indices) == expected_indices, f"Indices are not ranked correctly with metric={metric} and recency_bias={recency_bias}"

    def test_unknown_metric(self):
        """Test the behavior when an unknown metric is passed"""
        data_vectors = np.array([[1, 0], [0, 1]])
        query_vector = np.array([1, 0])
        with pytest.raises(ValueError):
            hyperDB_ranking_algorithm_sort(data_vectors, query_vector, metric='unknown_metric')

    def test_invalid_vector_shape(self):
        """Test the behavior with a non-2D vectors array for a non-cosine similarity metric"""
        data_vectors = np.array([1, 0])
        query_vector = np.array([1, 0])
        
        with pytest.raises(ValueError):
            # This should raise a ValueError since the data_vectors are not 2D
            hyperDB_ranking_algorithm_sort(data_vectors, query_vector, metric='euclidean_metric')

    def test_vectors_with_nan(self):
        """Test behavior when vectors contain NaN values"""
        data_vectors = np.array([[1, 0], [0, 1], [np.nan, np.nan]])
        query_vector = np.array([1, 0])
        
        # Executing the ranking algorithm
        with pytest.raises(ValueError):  # Assuming the function raises a ValueError for NaN
            top_indices, _ = hyperDB_ranking_algorithm_sort(data_vectors, query_vector, metric='cosine_similarity')
