import pytest
import numpy as np
import copy
import os
import time
from hyperdb import HyperDB

# Sample documents
sample_docs = [{"name": "Abra", "shortname": "abra", "hp": 160, "info": {"id": 63, "type": "psychic", "weakness": "dark", "description": "Sleeps 18 hours a day. If it senses danger, it will teleport itself to safety even as it sleeps."}, "images": {"photo": "images/abra.jpg", "typeIcon": "icons/psychic.jpg", "weaknessIcon": "icons/dark.jpg"}, "moves": [{"name": "Double Team", "type": "normal"}, {"name": "Energy Ball", "dp": 90, "type": "grass"}, {"name": "Psychic", "dp": 90, "type": "psychic"}, {"name": "Thief", "dp": 60, "type": "dark"}]},
{"name": "Aerodactyl", "shortname": "aerodactyl", "hp": 270, "info": {"id": 142, "type": "flying", "weakness": "water", "description": "This vicious Pokemon is said to have flown in ancient skies while shrieking high-pitched cries."}, "images": {"photo": "images/aerodactyl.jpg", "typeIcon": "icons/flying.jpg", "weaknessIcon": "icons/water.jpg"}, "moves": [{"name": "Bite", "dp": 60, "type": "dark"}, {"name": "Double Team", "type": "normal"}, {"name": "Ice Fang", "dp": 65, "type": "ice"}, {"name": "Wing Attack", "dp": 60, "type": "flying"}]},
{"name": "Alakazam", "shortname": "alakazam", "hp": 220, "info": {"id": 65, "type": "psychic", "weakness": "dark", "description": "Its brain can outperform a supercomputer. Its intelligence quotient is said to be 5,000."}, "images": {"photo": "images/alakazam.jpg", "typeIcon": "icons/psychic.jpg", "weaknessIcon": "icons/dark.jpg"}, "moves": [{"name": "Focus Blast", "dp": 120, "type": "fighting"}, {"name": "Kinesis", "type": "psychic"}, {"name": "Psychic", "dp": 90, "type": "psychic"}, {"name": "Shadow Ball", "dp": 80, "type": "ghost"}]},
{"name": "Arbok", "shortname": "arbok", "hp": 230, "info": {"id": 24, "type": "poison", "weakness": "psychic", "description": "It is rumored that the ferocious warning markings on its belly differ from area to area."}, "images": {"photo": "images/arbok.jpg", "typeIcon": "icons/poison.jpg", "weaknessIcon": "icons/psychic.jpg"}, "moves": [{"name": "Acid", "dp": 40, "type": "poison"}, {"name": "Bite", "dp": 60, "type": "dark"}, {"name": "Screech", "type": "normal"}, {"name": "Thunder Fang", "dp": 65, "type": "electric"}]},
{"name": "Arcanine", "shortname": "arcanine", "hp": 290, "info": {"id": 59, "type": "fire", "weakness": "water", "description": "This legendary Chinese Pokemon is considered magnificent. Many people are enchanted by its grand mane."}, "images": {"photo": "images/arcanine.jpg", "typeIcon": "icons/fire.jpg", "weaknessIcon": "icons/water.jpg"}, "moves": [{"name": "Bite", "dp": 60, "type": "dark"}, {"name": "Double Team", "type": "normal"}, {"name": "Extreme Speed", "dp": 80, "type": "normal"}, {"name": "Fire Fang", "dp": 65, "type": "fire"}]}]

# Sample vectors (for testing only)
sample_vectors = np.array([
    np.full(384, 1),  # Corresponding to "Abra"
    np.full(384, 2),  # Corresponding to "Aerodactyl"
    np.full(384, 3),  # Corresponding to "Alakazam"
    np.full(384, 4),  # Corresponding to "Arbok"
    np.full(384, 5),  # Corresponding to "Arcanine"
])

@pytest.fixture(scope='function')
def setup_db():
    # Initialize HyperDB instance with sample_docs and sample_vectors
    db = HyperDB(documents=copy.deepcopy(sample_docs), vectors=copy.deepcopy(sample_vectors), metadata_keys=['info.type'])
    #print(f"Debug: {len(db.vectors)} vectors, {len(db.source_indices)} source_indices")
    db._build_ann_index()
    return db

## Initialization and configuration tests
# Test to ensure query vector and database vectors have the same shape
def test_vector_shape(setup_db):
    query_vector = setup_db._generate_and_validate_query_vector("Abra")
    assert query_vector.ndim == 1, f"Dimension mismatch: query_vector.ndim = {query_vector.ndim}, expected 1"
    assert query_vector.shape[0] == setup_db.vectors.shape[1], f"Shape mismatch: query_vector.shape[0] = {query_vector.shape[0]}, setup_db.vectors.shape[1] = {setup_db.vectors.shape[1]}"

# Test to check floating-point precision
@pytest.mark.parametrize("fp_precision, expected_dtype", [
    ("float16", np.float16),
    ("float32", np.float32),
    ("float64", np.float64),
])
def test_floating_point_precision(fp_precision, expected_dtype):
    db = HyperDB(fp_precision=fp_precision)
    assert db.fp_precision == expected_dtype, f"The floating point precision should be set to {expected_dtype}"
    
    # If the vectors are not None, check their dtype
    if db.vectors is not None:
        assert db.vectors.dtype == expected_dtype, f"Vectors should have dtype {expected_dtype}, but got {db.vectors.dtype}"

# Test to check select_keys
@pytest.fixture(scope='function')
def setup_db_with_select_keys():
    select_keys = ["name", "info.type"]
    db = HyperDB(documents=copy.deepcopy(sample_docs), vectors=copy.deepcopy(sample_vectors), select_keys=select_keys)
    db._build_ann_index()
    return db

def test_select_keys_query_string(setup_db_with_select_keys):
    db = setup_db_with_select_keys
    query_input = "Abra psychic"  # A string query that would match the document with name "Abra" and info.type "psychic"
    results = db.query(query_input, top_k=1)
    print(results)
    assert len(results) == 1
    # Check that only the keys in 'select_keys' are present in the documents
    assert all(doc["info.type"] == "psychic" for doc, _ in results)

def test_select_keys_add(setup_db_with_select_keys):
    db = setup_db_with_select_keys
    new_doc = {"name": "Abra", "shortname": "abra", "hp": 160, "info": {"id": 63, "type": "psychic", "weakness": "dark", "description": "Sleeps 18 hours a day. If it senses danger, it will teleport itself to safety even as it sleeps."}, "images": {"photo": "images/abra.jpg", "typeIcon": "icons/psychic.jpg", "weaknessIcon": "icons/dark.jpg"}, "moves": [{"name": "Double Team", "type": "normal"}, {"name": "Energy Ball", "dp": 90, "type": "grass"}, {"name": "Psychic", "dp": 90, "type": "psychic"}, {"name": "Thief", "dp": 60, "type": "dark"}]}
    db.add([new_doc])
    assert len(db.documents) == len(sample_docs) + 1
    last_doc = db.documents[-1]
    # Check that only the keys in 'select_keys' are present in the newly added document
    expected_doc = {"name": "Abra", "info.type": "psychic"}
    print(f"Last-doc: {last_doc}")
    # Check that only the keys in 'select_keys' are present in the newly added document
    assert last_doc == expected_doc

# Test to check metadata_keys
@pytest.mark.parametrize("metadata_keys,expected", [
    (["info.type", "info.weakness"], True),
    (["info.id", "info.description"], True),
    (["images.photo", "images.typeIcon", "images.weaknessIcon"], True),
    (["moves"], True),  # Check for a list
    (["moves[0].name", "moves[0].type"], True),  # Check for keys within a list of dictionaries
    (["moves[0].dp"], True),  # Check for a key within a list of dictionaries that is only present in some of the dictionaries
    (["nonexistent.key"], False),  # Check for a nonexistent key
    ([], True),  # Check for an empty list of keys
])
def test_metadata_keys(metadata_keys, expected):
    try:
        db = HyperDB(documents=sample_docs, metadata_keys=metadata_keys)
        # If no exception is raised, assert that we expected this to be True
        assert expected is True, f"Metadata keys check failed for {metadata_keys}. Expected {expected} but got True"
    except ValueError as e:
        # If a ValueError is raised, assert that we expected this to be False
        assert expected is False, f"Metadata keys check failed for {metadata_keys}. Expected {expected} but got False"

## Timestamp when add_timestamp is set
# Test to ensure that timestamps are added when add_timestamp=True
def test_add_timestamp(setup_db):
    db = HyperDB(add_timestamp=True)
    new_doc = {"name": "Pikachu", "shortname": "pikachu", "hp": 160, "info": {"id": 25, "type": "electric", "weakness": "ground", "description": "Melissa's favorite Pokemon! When several Pikachu gather, their electricity could build and cause lightning storms."}, "images": {"photo": "images/pikachu.jpg", "typeIcon": "icons/electric.jpg", "weaknessIcon": "icons/ground.jpg"}, "moves": [{"name": "Growl", "type": "normal"}, {"name": "Quick Attack", "dp": 40, "type": "normal"}, {"name": "Thunderbolt", "dp": 90, "type": "electric"}]}
    db.add(new_doc)
    assert "timestamp" in db._metadata_index[len(db.documents) - 1], "Timestamp should be present in the metadata"

# Test to ensure that adding a document with add_timestamp adds a timestamp and it's near the current time
def test_add_document_with_timestamp(setup_db):
    db = HyperDB(add_timestamp=True)
    new_doc = {"name": "Mewtwo", "shortname": "mewtwo", "hp": 160, "info": {"id": 150, "type": "psychic", "weakness": "dark", "description": "It was created by a scientist after years of horrific gene splicing and DNA engineering experiments."}, "images": {"photo": "images/mewtwo.jpg", "typeIcon": "icons/psychic.jpg", "weaknessIcon": "icons/dark.jpg"}, "moves": [{"name": "Amnesia", "type": "psychic"}, {"name": "Psychic", "dp": 90, "type": "psychic"}, {"name": "Swift", "dp": 60, "type": "normal"}]}
    db.add(new_doc)
    current_time = time.time()
    last_document_index = len(db.documents) - 1
    assert abs(db._metadata_index[last_document_index]["timestamp"] - current_time) < 5, "Timestamp should be within 5 seconds of the current time"

## Document addition and removal tests
# Test to ensure that adding a single document works as expected
def test_add_single_document(setup_db):
    db = setup_db
    new_doc = {"name": "Charizard", "hp": 300, "info": {"type": "fire", "weakness": "water"}}
    db.add(new_doc)
    assert len(db.documents) == 6, f"Expected 6 documents, but got {len(db.documents)}"

# Test to ensure that adding multiple documents at once works as expected
def test_add_multiple_documents(setup_db):
    db = setup_db
    new_docs = [
        {"name": "Blastoise", "shortname": "blastoise", "hp": 268, "info": {"id": 9, "type": "water", "weakness": "grass", "description": "The rocket cannons on its shell fire jets of water capable of punching holes through thick steel."}, "images": {"photo": "images/blastoise.jpg", "typeIcon": "icons/water.jpg", "weaknessIcon": "icons/grass.jpg"}, "moves": [{"name": "Bite", "dp": 60, "type": "dark"}, {"name": "Flash Cannon", "dp": 80, "type": "steel"}, {"name": "Hydro Pump", "dp": 110, "type": "water"}, {"name": "Withdraw", "type": "water"}]},
        {"name": "Venusaur", "shortname": "venusaur", "hp": 160, "info": {"id": 3, "type": "grass", "weakness": "fire", "description": "The plant blooms when it is absorbing solar energy. It stays on the move to seek sunlight."}, "images": {"photo": "images/venusaur.jpg", "typeIcon": "icons/grass.jpg", "weaknessIcon": "icons/fire.jpg"}, "moves": [{"name": "Razor Leaf", "dp": 55, "type": "grass"}]}
    ]
    db.add(new_docs)
    assert len(db.documents) == 7, f"Expected 7 documents, but got {len(db.documents)}"

# Test to ensure that removing a single document works as expected
def test_remove_single_document(setup_db):
    db = setup_db
    db.remove_document(0)  # Remove the first document
    assert len(db.documents) == 4, f"Expected 4 documents, but got {len(db.documents)}"

# Test to ensure that removing multiple documents at once works as expected
def test_remove_multiple_documents(setup_db):
    db = setup_db
    db.remove_document([0, 1])  # Remove the first and second documents
    assert len(db.documents) == 3, f"Expected 3 documents, but got {len(db.documents)}"

## Database information tests
# Test to ensure the `size` method returns the correct number of documents
def test_size(setup_db):
    db = setup_db
    
    # Test size for initialized db
    initial_size = len(sample_docs)  # sample_docs is the list of initial documents
    assert db.size() == initial_size, f"Expected initial size to be {initial_size}, but got {db.size()}"
    
    # Test size when adding a single document
    new_doc = {"name": "Pikachu", "shortname": "pikachu", "hp": 160, "info": {"id": 25, "type": "electric", "weakness": "ground", "description": "Melissa's favorite Pokemon! When several Pikachu gather, their electricity could build and cause lightning storms."}, "images": {"photo": "images/pikachu.jpg", "typeIcon": "icons/electric.jpg", "weaknessIcon": "icons/ground.jpg"}, "moves": [{"name": "Growl", "type": "normal"}, {"name": "Quick Attack", "dp": 40, "type": "normal"}, {"name": "Thunderbolt", "dp": 90, "type": "electric"}]}

    db.add(new_doc)
    assert db.size() == initial_size + 1, f"Expected size to be {initial_size + 1}, but got {db.size()}"
    
    # Test size when removing a document
    db.remove_document(0)
    assert db.size() == initial_size, f"Expected size to be back to {initial_size}, but got {db.size()}"
    
    # Test size with chunks (assuming chunks would just duplicate documents for this example)
    assert db.size(with_chunks=True) == initial_size, f"Expected size with chunks to be {initial_size}, but got {db.size(with_chunks=True)}"
    
    # Test size with an empty db
    empty_db = HyperDB()
    assert empty_db.size() == 0, f"Expected size of an empty database to be 0, but got {empty_db.size()}"

# Test to ensure the `dict` method works as expected without vectors
def test_dict_without_vectors(setup_db):
    db = setup_db
    db_dict = db.dict(vectors=False)
    assert len(db_dict) == 5, f"Expected 5 entries in the dictionary, but got {len(db_dict)}"

# Test to ensure the `dict` method works as expected with vectors
def test_dict_with_vectors(setup_db):
    db = setup_db
    db_dict = db.dict(vectors=True)
    assert len(db_dict) == 5, f"Expected 5 entries in the dictionary, but got {len(db_dict)}"
    for entry in db_dict:
        assert 'vector' in entry, "Expected 'vector' key in each entry"

### Querying tests
# Test if an empty database raises an Exception during a query
def test_query_empty_db():
    empty_db = HyperDB()
    with pytest.raises(Exception):
        empty_db.query("Abra")

# Test if different types of query inputs are supported
@pytest.mark.parametrize("query_input", [("Abra"), ("Abra"), (np.random.rand(384)), (np.random.rand(384))])
def test_query_input_types(setup_db, query_input):
    results = setup_db.query(query_input)
    assert len(results) > 0, f"Query should return results for input: {query_input}"

@pytest.mark.parametrize("top_k, expected_length", [(3, 3), (3, 3), (10, 5), (10, 5), (1, 1), (1, 1)])
def test_query_top_k(setup_db, top_k, expected_length):
    results = setup_db.query("Abra", top_k=top_k)
    assert len(results) == expected_length, f"Expected {expected_length} results, but got {len(results)}."

# Test if the 'return_similarities' parameter affects the type of the returned results
@pytest.mark.parametrize("return_similarities, result_type", [(True, tuple), (False, dict)])
def test_query_return_similarities(setup_db, return_similarities, result_type):
    results = setup_db.query("Abra", return_similarities=return_similarities)
    assert all(isinstance(r, result_type) for r in results), f"All results should be of type {result_type}"

# Test if filters can be applied during the query
def test_query_with_filters(setup_db):
    filters = [("key", "name"), ("metadata", {"info.type": "psychic"})]
    for use_ann in [True, False]:
        filtered_results = setup_db.query("Abra", filters=filters)
        assert all(r['info']['type'] == 'psychic' for r, _ in filtered_results), f'All results should have info.type as "psychic" with use_ann={use_ann}'

# Test if multiple filters can be applied with the query
@pytest.fixture
def metadata_keys(request):
    return request.param

@pytest.fixture
def setup_db_with_metadata(metadata_keys):
    db = HyperDB(
        documents=copy.deepcopy(sample_docs),
        vectors=copy.deepcopy(sample_vectors),
        metadata_keys=metadata_keys
    )
    db._build_ann_index()
    return db

@pytest.mark.parametrize("metadata_keys", [['info.type', 'info.weakness', 'moves[0].name', 'info.id']], indirect=True)
@pytest.mark.parametrize("filters,expected", [
    # Test 1: Check for a single metadata filter
    (
        [("metadata", {"info.type": "psychic"})],
        lambda r: all(doc['info']['type'] == 'psychic' for doc, _ in r)
    ),
    # Test 2: Check for multiple metadata filters
    (
        [("metadata", {"info.type": "psychic", "info.weakness": "dark"})],
        lambda r: all(doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' for doc, _ in r)
    ),
    # Test 3: Check for a single key filter
    (
        [("key", "name")],
        lambda r: all('name' in doc for doc, _ in r)
    ),
    # Test 4: Check for multiple key filters
    (
        [("key", ["name", "info.description"])],
        lambda r: all('name' in doc and doc['info']['description'] for doc, _ in r)
    ),
    # Test 5: Check for a single sentence filter
    (
        [("sentence", ["Sleeps 18 hours a day"])],
        lambda r: all('Sleeps 18 hours a day' in doc['info']['description'] for doc, _ in r)
    ),
    # Test 6: Check for multiple sentence filters
    (
        [("sentence", ["Sleeps 18 hours a day", "teleport itself to safety"])],
        lambda r: any('Sleeps 18 hours a day' in doc['info']['description'] or 'teleport itself to safety' in doc['info']['description'] for doc, _ in r)
    ),
    # Test 7: Check for mixed filters
    (
        [("metadata", {"info.type": "psychic"}), ("key", "moves"), ("sentence", ["Sleeps 18 hours a day"])],
        lambda r: all(doc['info']['type'] == 'psychic' and 'moves' in doc and 'Sleeps 18 hours a day' in doc['info']['description'] for doc, _ in r)
    ),
    # Test 8: Mixed filters with multiple keys and metadata
    (
        [("key", ["name", "info.description"]), ("metadata", {"info.type": "psychic", "info.weakness": "dark"})],
        lambda r: all('name' in doc and doc['info']['description'] and doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' for doc, _ in r)
    ),
    # Test 9: Mixed filters with multiple keys, metadata and sentences
    (
        [("key", ["name", "info.description"]), ("metadata", {"info.type": "psychic", "info.weakness": "dark"}), ("sentence", ["Sleeps 18 hours a day", "teleport itself to safety"])],
        lambda r: all(('name' in doc and doc['info']['description'] and doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' and ('Sleeps 18 hours a day' in doc['info']['description'] or 'teleport itself to safety' in doc['info']['description'])) for doc, _ in r)
    ),
    # Test 10: Mixed filters with nested array keys
    (
        [("key", ["moves[0].name", "moves[0].type"]), ("metadata", {"info.type": "psychic"})],
        lambda r: all('moves' in doc and doc['moves'][0]['name'] is not None and doc['moves'][0]['type'] is not None and doc['info']['type'] == 'psychic' for doc, _ in r)
    ),
    # Test 11: Mixed filters with deeply nested keys
    (
        [("key", ["moves[1].name", "moves[1].type", "moves[1].dp"]), ("metadata", {"info.type": "psychic", "info.id": 63})],
        lambda r: all(('moves' in doc and doc['moves'][1]['name'] is not None and doc['moves'][1]['type'] is not None and doc['moves'][1].get('dp', None) is not None and doc['info']['type'] == 'psychic' and doc['info']['id'] == 63) for doc, _ in r)
    )
])
def test_query_multiple_filters(setup_db_with_metadata, filters, expected):
    results = setup_db_with_metadata.query("Query Text", filters=filters)
    print(results)  # Add this line for debugging
    assert expected(results), f'All results should meet filter criteria: {filters}'

# Test if invalid filters raise an Exception during the query
def test_query_invalid_filters(setup_db):
    filters = [("key", "invalid_key"), ("metadata", {"info.invalid_key": "psychic"})]
    with pytest.raises(Exception):
        setup_db.query("Abra", filters=filters)

# Test if an invalid metric raises a ValueError during the query
def test_query_invalid_metric(setup_db):
    with pytest.raises(ValueError):
        setup_db.query("Abra", metric="invalid_metric")

# Test if recency bias can be applied during the query with a dummy timestamp
def test_query_with_recency_bias(setup_db):
    setup_db.metadata_keys.append('hp')
    print(f"Metadata keys: {setup_db.metadata_keys}")
    recency_results = setup_db.query("Abra", recency_bias=1, timestamp_key='hp')
    assert recency_results[0][0]['name'] == 'Arcanine', f"The first result should be 'Arcanine', but got {recency_results[0][0]['name']}"

# Test if the query handles negative recency_bias correctly
def test_query_negative_recency_bias(setup_db):
    # Add a significantly spaced timestamp field to sample documents
    for i, doc in enumerate(setup_db.documents):
        doc['timestamp'] = i
    setup_db.metadata_keys.append('timestamp')
    
    results = setup_db.query("Abra", recency_bias=-1, timestamp_key='timestamp')
    assert results[0][0]['name'] == 'Abra', f"The first result should be 'Abra', but got {results[0][0]['name']}"

# Test if recency bias uses the default 'timestamp' key when timestamp_key is None
def test_query_default_timestamp_key(setup_db):
    # Add a timestamp field to sample documents
    for i, doc in enumerate(setup_db.documents):
        doc['timestamp'] = i
    setup_db.metadata_keys.append('timestamp')
    
    results = setup_db.query("Abra", recency_bias=1, timestamp_key=None)
    assert results[0][0]['name'] == 'Arcanine', f"The first result should be 'Arcanine', but got {results[0][0]['name']}"

# Test if the query raises an error when recency_bias is not zero and timestamp_key is None but there's no 'timestamp' key
def test_query_no_default_timestamp_key(setup_db):
    with pytest.raises(ValueError):
        setup_db.query("Abra", recency_bias=1)

# Test if ANN pre-filtering logic works correctly
def test_query_with_ann_prefilter(setup_db):
    # Use a metric compatible with ANN (e.g., 'cosine_similarity')
    ann_results = setup_db.query("Abra", metric='cosine_similarity')
    assert len(ann_results) > 0, "ANN pre-filtering should return results."

    # Use a metric not compatible with ANN (e.g., 'pearson_correlation')
    non_ann_results = setup_db.query("Abra", metric='pearson_correlation')
    assert len(non_ann_results) > 0, "Non-ANN metric should also return results."

# Test if the recency bias logic works correctly
@pytest.mark.parametrize("recency_bias, expected_first", [(1, 'Arcanine'), (-1, 'Abra')])
def test_query_with_recency_bias(setup_db, recency_bias, expected_first):
    setup_db.metadata_keys.append('hp')
    recency_results = setup_db.query("Abra", recency_bias=recency_bias, timestamp_key='hp')
    assert recency_results[0][0]['name'] == expected_first, f"The first result should be '{expected_first}', but got {recency_results[0][0]['name']}"

# Test if timestamp logic works correctly
def test_query_with_timestamp_key(setup_db):
    setup_db.metadata_keys.append('timestamp')
    for i, doc in enumerate(setup_db.documents):
        doc['timestamp'] = i

    # Test with a positive recency bias
    results = setup_db.query("Abra", recency_bias=1, timestamp_key='timestamp')
    assert results[0][0]['name'] == 'Arcanine', f"The first result should be 'Arcanine', but got {results[0][0]['name']}"

    # Test with a negative recency bias
    results = setup_db.query("Abra", recency_bias=-1, timestamp_key='timestamp')
    assert results[0][0]['name'] == 'Abra', f"The first result should be 'Abra', but got {results[0][0]['name']}"

# Test if missing timestamp_key raises an error
def test_query_missing_timestamp_key(setup_db):
    with pytest.raises(ValueError):
        setup_db.query("Abra", recency_bias=1, timestamp_key='missing_timestamp')

# Test if inconsistent ANN and brute-force results raise an error
def test_query_fallback_to_bruteforce(setup_db, capsys):
    # Introduce an inconsistency between ANN and brute-force results (for testing purposes)
    setup_db.documents.append({'name': 'TestDoc'})
    setup_db.vectors = np.vstack([setup_db.vectors, np.random.rand(1, 384)])
    setup_db._build_ann_index()

    setup_db.query("Abra", metric='pearson_correlation')
    captured = capsys.readouterr()
    assert "Bruteforce method used instead" in captured.out, "Should fallback to brute-force when using an incompatible metric."

# Test if query handles empty result set after applying filters
def test_query_empty_after_filters(setup_db):
    filters = [("metadata", {"info.type": "non_existent_type"})]
    results = setup_db.query("Abra", filters=filters)
    assert len(results) == 0, "Query with filters that result in an empty set should return an empty list."

## Database Saving and Loading Tests
# Test for invalid format in save
def test_save_invalid_format(setup_db, tmp_path):
    db = setup_db
    file_path = str(tmp_path / "test_invalid.xyz")
    with pytest.raises(ValueError):
        db.save(file_path, format='xyz')
        
# Test for save method with pickle format
def test_save_pickle(setup_db, tmp_path):
    db = setup_db
    file_path = str(tmp_path / "test_save_pickle.pkl")
    db.save(file_path, format='pickle')

    assert os.path.exists(file_path)

    new_db = HyperDB()
    new_db.load(file_path, format='pickle')

    # Check data integrity
    assert new_db.documents == db.documents
    assert np.array_equal(new_db.vectors, db.vectors)
    assert new_db.source_indices == db.source_indices
    assert new_db._metadata_index == db._metadata_index
    assert new_db.split_info == db.split_info

# Test for save method with JSON format
def test_save_json(setup_db, tmp_path):
    db = setup_db
    file_path = str(tmp_path / "test_save_json.json")
    db.save(file_path, format='json')

    assert os.path.exists(file_path)

    new_db = HyperDB()
    new_db.load(file_path, format='json')

    # Check data integrity
    assert new_db.documents == db.documents
    assert np.array_equal(new_db.vectors, db.vectors)
    assert new_db.source_indices == db.source_indices
    assert new_db._metadata_index == db._metadata_index
    assert new_db.split_info == db.split_info

# Test for save method with SQLite format
def test_save_sqlite(setup_db, tmp_path):
    db = setup_db
    file_path = str(tmp_path / "test_save_sqlite.db")
    db.save(file_path, format='sqlite')

    assert os.path.exists(file_path)

    new_db = HyperDB()
    new_db.load(file_path, format='sqlite')

    # Check data integrity
    assert new_db.documents == db.documents
    assert np.array_equal(new_db.vectors, db.vectors)
    assert new_db.source_indices == db.source_indices
    assert new_db._metadata_index == db._metadata_index
    assert new_db.split_info == db.split_info

# Test for load method with pickle format
def test_load_pickle(setup_db, tmp_path):
    db = setup_db
    file_path = str(tmp_path / "test_load_pickle.pkl")
    db.save(file_path, format='pickle')

    # Load into a new database
    new_db = HyperDB()
    new_db.load(file_path, format='pickle')

    # Check document count
    assert len(new_db.documents) == len(db.documents)

    # Check data integrity
    assert new_db.documents == db.documents
    assert np.array_equal(new_db.vectors, db.vectors)
    assert new_db.source_indices == db.source_indices
    assert new_db._metadata_index == db._metadata_index
    assert new_db.split_info == db.split_info

# Test for load method with JSON format
def test_load_json(setup_db, tmp_path):
    db = setup_db
    file_path = tmp_path / "test_load_json.json"
    db.save(file_path, format='json')

    # Load into a new database
    new_db = HyperDB()
    new_db.load(file_path, format='json')

    # Check document count
    assert len(new_db.documents) == len(db.documents)

    # Check data integrity
    assert new_db.documents == db.documents
    assert np.array_equal(new_db.vectors, db.vectors)
    assert new_db.source_indices == db.source_indices
    assert new_db._metadata_index == db._metadata_index
    assert new_db.split_info == db.split_info

# Test for load method with SQLite format
def test_load_sqlite(setup_db, tmp_path):
    db = setup_db
    file_path = tmp_path / "test_load_sqlite.db"
    db.save(file_path, format='sqlite')

    # Load into a new database
    new_db = HyperDB()
    new_db.load(file_path, format='sqlite')

    # Check document count
    assert len(new_db.documents) == len(db.documents)

    # Check data integrity
    assert new_db.documents == db.documents
    assert np.array_equal(new_db.vectors, db.vectors)
    assert new_db.source_indices == db.source_indices
    assert new_db._metadata_index == db._metadata_index
    assert new_db.split_info == db.split_info

## Additional functionality
# Test for compute_and_save_word_frequencies
def test_compute_and_save_word_frequencies(setup_db, tmp_path):
    db = setup_db
    file_path = tmp_path / "word_frequencies.txt"
    db.compute_and_save_word_frequencies(file_path)
    assert file_path.exists()
    with open(file_path, 'r') as f:
        content = f.read()
    assert "abra" in content.lower()
