import pytest
import numpy as np
import copy
import os
import time
import cachetools
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
    assert all(doc["info.type"] == "psychic" for doc, _, _ in results)

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
def test_add_timestamp():
    db = HyperDB(add_timestamp=True)
    new_doc = {"name": "Pikachu", "shortname": "pikachu", "hp": 160, "info": {"id": 25, "type": "electric", "weakness": "ground", "description": "Melissa's favorite Pokemon! When several Pikachu gather, their electricity could build and cause lightning storms."}, "images": {"photo": "images/pikachu.jpg", "typeIcon": "icons/electric.jpg", "weaknessIcon": "icons/ground.jpg"}, "moves": [{"name": "Growl", "type": "normal"}, {"name": "Quick Attack", "dp": 40, "type": "normal"}, {"name": "Thunderbolt", "dp": 90, "type": "electric"}]}
    db.add(new_doc)
    assert "timestamp" in db._metadata_index[len(db.documents) - 1], "Timestamp should be present in the metadata"

# Test to ensure that adding a document with add_timestamp adds a timestamp and it's near the current time
def test_add_document_with_timestamp():
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

# Test to ensure that adding multiple documents consecutively works as expected
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

## Check consistency with split_info and source_indices for large documents > 510 tokens
# Test to ensure that `add_document` method handles properly split_info and source_indices for large documents
def test_add_chunked_document():
    setup_db = HyperDB()
    # Simulate a large document
    large_doc = {"text": "word " * 700}  # Assuming each chunk can be 510 tokens max

    # Adding the large document to the database
    setup_db.add(large_doc)

    # The expected number of chunks for the large document
    expected_chunks = 2  # Since it's more than 510 tokens this would be split into 2 chunks

    # Check if there is a correct number of documents and vectors
    expected_doc_count = 1
    assert len(setup_db.documents) == expected_doc_count, "Incorrect number of documents after simple addition"

    expected_vectors_count = 2
    assert len(setup_db.vectors) == expected_vectors_count, "Incorrect number of vectors after simple addition"

    # Check if split_info is updated correctly
    new_doc_index = len(setup_db.documents) - 1
    assert setup_db.split_info[new_doc_index] == expected_chunks, "split_info not updated correctly for the chunked document"

    # Check if source_indices are updated correctly
    chunk_indices = [i for i, idx in enumerate(setup_db.source_indices) if idx == new_doc_index]
    assert len(chunk_indices) == expected_chunks, "source_indices not updated correctly for the chunked document"

def test_add_multiple_documents_with_chunking():
    # Clear existing state in the database to ensure test independence
    setup_db = HyperDB()

    # Simulate multiple documents, some are large and some are not
    large_doc1 = {"text": "word " * 600}  # Large enough to be split
    large_doc2 = {"text": "word " * 700}  # Large enough to be split
    regular_doc = {"text": "word " * 400}  # Not large enough to be split
    documents = [large_doc1, large_doc2, regular_doc]

    # Adding the documents to the database
    setup_db.add(documents)

    # Assuming the embedding function will split large documents into 2 chunks each
    expected_chunks_large_doc1 = 2
    expected_chunks_large_doc2 = 2
    expected_chunks_regular_doc = 1

    # Retrieve the indices in the database, corresponding to the newly added documents
    doc_indices = range(0, 3)

    # Check if there is a correct number of documents and vectors
    expected_doc_count = 3
    assert len(setup_db.documents) == expected_doc_count, "Incorrect number of documents after multiple addition"

    expected_vectors_count = 5
    assert len(setup_db.vectors) == expected_vectors_count, "Incorrect number of vectors after multiple addition"

    # Check if split_info is updated correctly for each document
    assert setup_db.split_info[doc_indices[0]] == expected_chunks_large_doc1, "Incorrect split_info for the first large document"
    assert setup_db.split_info[doc_indices[1]] == expected_chunks_large_doc2, "Incorrect split_info for the second large document"
    assert setup_db.split_info[doc_indices[2]] == expected_chunks_regular_doc, "Incorrect split_info for the regular document"

    # Check if source_indices are updated correctly for each document
    for idx, expected_chunks in zip(doc_indices, [expected_chunks_large_doc1, expected_chunks_large_doc2, expected_chunks_regular_doc]):
        chunk_indices = [i for i, source_idx in enumerate(setup_db.source_indices) if source_idx == idx]
        assert len(chunk_indices) == expected_chunks, f"Incorrect source_indices for document at index {idx}"

# Test to ensure that `remove_document` method handles properly split_info and source_indices for large documents
def test_remove_chunked_document():
    setup_db = HyperDB()
    # Add and then remove a large document
    large_doc = {"text": "word " * 600}  # Simulated large document
    setup_db.add(large_doc)
    new_doc_index = len(setup_db.documents) - 1
    setup_db.remove_document(new_doc_index)

    # Assertions to check if the database state is coherent after removal
    assert not setup_db.documents, "Document not removed correctly"

    assert setup_db.vectors.size == 0, "Vectors not removed correctly"

    # Check if split_info is updated correctly
    assert new_doc_index not in setup_db.split_info, "split_info not updated correctly after removing the chunked document"

    # Check if source_indices are updated correctly
    assert not any(idx == new_doc_index for idx in setup_db.source_indices), "source_indices not updated correctly after removing the chunked document"

def test_remove_large_document():
    setup_db = HyperDB()

    # Add a large document
    large_doc = {"text": "word " * 600}
    setup_db.add(large_doc)

    # Remove the large document
    setup_db.remove_document(0)  # Assuming it's the first document

    # Assertions to check if the database state is coherent after removal
    assert not setup_db.documents, "Document not removed correctly"

    assert setup_db.vectors.size == 0, "Vectors not removed correctly"

    # Check if split_info is updated correctly
    assert not setup_db.split_info, "split_info not updated correctly after removing the chunked document"

    # Check if source_indices are updated correctly
    assert not setup_db.source_indices, "source_indices not updated correctly after removing the chunked document"

def test_remove_large_document_among_multiple():
    setup_db = HyperDB()

    # Add documents: regular, large, regular
    regular_doc1 = {"text": "word " * 400}
    large_doc = {"text": "word " * 700}  # Large enough to be chunked (2 chunks)
    regular_doc2 = {"text": "word " * 400}

    setup_db.add([regular_doc1, large_doc, regular_doc2])

    print(f"Test docs: {len(setup_db.documents)}")
    # Remove the large document
    setup_db.remove_document(1)  # Assuming it's the second document
    print(f"Test docs after remove: {len(setup_db.documents)}")

    # Check if the number of documents and vectors are correct after removal
    expected_doc_count = 2  # One large document (2 chunks) removed, 2 small documents remain
    assert len(setup_db.documents) == expected_doc_count, "Incorrect number of documents after removal"

    expected_vectors_count = 2 # Two regular docs = 2 vectors
    assert len(setup_db.vectors) == expected_vectors_count, "Incorrect number of vectors after removal"

    # Check if split_info is updated correctly
    expected_split_info = {0: 1, 1: 1}
    assert setup_db.split_info == expected_split_info, "split_info not updated correctly after removal"

    # Check if source_indices are updated correctly
    expected_source_indices = [0, 1]
    assert setup_db.source_indices == expected_source_indices, "source_indices not updated correctly after removal"

def test_remove_large_document_among_multiple_bis():
    setup_db = HyperDB()

    # Add documents: regular, large, regular
    regular_doc1 = {"text": "word " * 400}
    large_doc = {"text": "word " * 700}  # Large enough to be chunked (2 chunks)
    regular_doc2 = {"text": "word " * 400}
    large_doc2 = {"text": "word " * 700}  # Large enough to be chunked (2 chunks)

    setup_db.add([regular_doc1, large_doc, regular_doc2, large_doc2])

    # Remove the large document
    setup_db.remove_document(1)  # Assuming it's the second document

    # Check if the number of documents is correct after removal
    expected_doc_count = 3  # One large document (2 chunks) removed, 2 small documents remain
    assert len(setup_db.documents) == expected_doc_count, "Incorrect number of documents after removal"

    expected_vectors_count = 4
    assert len(setup_db.vectors) == expected_vectors_count, "Incorrect number of vectors after removal"

    # Check if split_info is updated correctly
    expected_split_info = {0: 1, 1: 1, 2: 2}
    assert setup_db.split_info == expected_split_info, "split_info not updated correctly after removal"

    # Check if source_indices are updated correctly
    expected_source_indices = [0, 1, 2, 2]
    assert setup_db.source_indices == expected_source_indices, "source_indices not updated correctly after removal"

# Test to ensure that `save` method handles properly split_info and source_indices after adding a large document
def test_add_chunked_document_with_save_and_load(setup_db, tmp_path):
    setup_db = HyperDB()
    # Simulate a large document
    large_doc = {"text": "word " * 600}  # Assuming each chunk can be 510 tokens max

    # Adding the large document to the database
    setup_db.add(large_doc)

    # Save the database state
    file_path = str(tmp_path / "db_save.pkl")
    setup_db.save(file_path, format='pickle')

    # Load the database from the saved state
    new_db = HyperDB()
    new_db.load(file_path, format='pickle')

    # The expected number of chunks for the large document
    expected_chunks = 2  # Since 600 words would be split into 2 chunks

    # Check if split_info is updated correctly in the loaded database
    new_doc_index = len(new_db.documents) - 1
    assert new_db.split_info[new_doc_index] == expected_chunks, "split_info not updated correctly in the loaded database"

    # Check if source_indices are updated correctly in the loaded database
    chunk_indices = [i for i, idx in enumerate(new_db.source_indices) if idx == new_doc_index]
    assert len(chunk_indices) == expected_chunks, "source_indices not updated correctly in the loaded database"

# Test to ensure that `save` method handles properly split_info and source_indices after removing a large document
def test_remove_chunked_document_with_save_and_load(setup_db, tmp_path):
    # Add and then remove a large document
    large_doc = {"text": "word " * 600}  # Simulated large document
    setup_db.add(large_doc)
    new_doc_index = len(setup_db.documents) - 1
    setup_db.remove_document(new_doc_index)

    # Save the database state
    file_path = str(tmp_path / "db_save.pkl")
    setup_db.save(file_path, format='pickle')

    # Load the database from the saved state
    new_db = HyperDB()
    new_db.load(file_path, format='pickle')

    # Check if split_info is updated correctly in the loaded database
    assert new_doc_index not in new_db.split_info, "split_info not updated correctly after removing the chunked document in the loaded database"

    # Check if source_indices are updated correctly in the loaded database
    assert not any(idx == new_doc_index for idx in new_db.source_indices), "source_indices not updated correctly after removing the chunked document in the loaded database"

## Test vectors uniformity
@pytest.mark.parametrize("test_input,expected_exception,expected_message,raises_exception", [
    # Test cases for valid inputs
    ([np.random.rand(128) for _ in range(10)], None, None, False),

    # Test cases for invalid inputs
    ([[1, 2, 3], [4, 5, 6, 7]], ValueError, "All vectors must have the same dimension.", True),
    (np.array([1, 2, 3, 4, 5, 6, 7, 8, 9]).reshape(3, 3, 1), ValueError, "Vectors do not have the expected structure.", True),
])
def test_vector_uniformity(test_input, expected_exception, expected_message, raises_exception):
    db = HyperDB()
    if raises_exception:
        with pytest.raises(expected_exception, match=expected_message):
            db.validate_vector_uniformity(test_input)
    else:
        db.validate_vector_uniformity(test_input)  # Should not raise an exception

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
        lambda r: all(doc['info']['type'] == 'psychic' for doc, _, _ in r)
    ),
    # Test 2: Check for multiple metadata filters
    (
        [("metadata", {"info.type": "psychic", "info.weakness": "dark"})],
        lambda r: all(doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' for doc, _, _ in r)
    ),
    # Test 3: Check for a single key filter
    (
        [("key", "name")],
        lambda r: all('name' in doc for doc, _, _ in r)
    ),
    # Test 4: Check for multiple key filters
    (
        [("key", ["name", "info.description"])],
        lambda r: all('name' in doc and doc['info']['description'] for doc, _, _ in r)
    ),
    # Test 5: Check for a single sentence filter
    (
        [("sentence", ["Sleeps 18 hours a day"])],
        lambda r: all('Sleeps 18 hours a day' in doc['info']['description'] for doc, _, _ in r)
    ),
    # Test 6: Check for multiple sentence filters
    (
        [("sentence", ["Sleeps 18 hours a day", "teleport itself to safety"])],
        lambda r: any('Sleeps 18 hours a day' in doc['info']['description'] or 'teleport itself to safety' in doc['info']['description'] for doc, _, _ in r)
    ),
    # Test 7: Check for mixed filters
    (
        [("metadata", {"info.type": "psychic"}), ("key", "moves"), ("sentence", ["Sleeps 18 hours a day"])],
        lambda r: all(doc['info']['type'] == 'psychic' and 'moves' in doc and 'Sleeps 18 hours a day' in doc['info']['description'] for doc, _, _ in r)
    ),
    # Test 8: Mixed filters with multiple keys and metadata
    (
        [("key", ["name", "info.description"]), ("metadata", {"info.type": "psychic", "info.weakness": "dark"})],
        lambda r: all('name' in doc and doc['info']['description'] and doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' for doc, _, _ in r)
    ),
    # Test 9: Mixed filters with multiple keys, metadata and sentences
    (
        [("key", ["name", "info.description"]), ("metadata", {"info.type": "psychic", "info.weakness": "dark"}), ("sentence", ["Sleeps 18 hours a day", "teleport itself to safety"])],
        lambda r: all(('name' in doc and doc['info']['description'] and doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' and ('Sleeps 18 hours a day' in doc['info']['description'] or 'teleport itself to safety' in doc['info']['description'])) for doc, _, _ in r)
    ),
    # Test 10: Mixed filters with nested array keys
    (
        [("key", ["moves[0].name", "moves[0].type"]), ("metadata", {"info.type": "psychic"})],
        lambda r: all('moves' in doc and doc['moves'][0]['name'] is not None and doc['moves'][0]['type'] is not None and doc['info']['type'] == 'psychic' for doc, _, _ in r)
    ),
    # Test 11: Mixed filters with deeply nested keys
    (
        [("key", ["moves[1].name", "moves[1].type", "moves[1].dp"]), ("metadata", {"info.type": "psychic", "info.id": 63})],
        lambda r: all(('moves' in doc and doc['moves'][1]['name'] is not None and doc['moves'][1]['type'] is not None and doc['moves'][1].get('dp', None) is not None and doc['info']['type'] == 'psychic' and doc['info']['id'] == 63) for doc, _, _ in r)
    ),
    # Test 12: Check for a single skip_doc filter (skip the first 2 documents)
    (
        [("skip_doc", 2)],
        lambda r: len(r) == len(sample_docs) - 2  # Expect two less documents
    ),
    # Test 13: Check for a single skip_doc filter (skip last 2 documents)
    (
        [("skip_doc", -2)],
        lambda r: len(r) == len(sample_docs) - 2  # Expect two less documents
    ),
    # Test 14: Check for a skip_doc filter combined with metadata filter
    (
        [("skip_doc", 2), ("metadata", {"info.type": "psychic"})],
        lambda r: all(doc['info']['type'] == 'psychic' for doc, _, _ in r) and len(r) <= len(sample_docs) - 2
    ),
    # Test 15: Check for a skip_doc filter combined with multiple filters
    (
        [("skip_doc", 1), ("key", ["name", "info.description"]), ("metadata", {"info.type": "psychic", "info.weakness": "dark"})],
        lambda r: all('name' in doc and doc['info']['description'] and doc['info']['type'] == 'psychic' and doc['info']['weakness'] == 'dark' for doc, _, _ in r) and len(r) <= len(sample_docs) - 1
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
    # Introduce a scenario where ANN-based query would fail
    # Example: Using a metric not compatible with ANN
    incompatible_metric = 'pearson_correlation'

    # Call the query method with the incompatible metric
    setup_db.query("Abra", metric=incompatible_metric)

    # Capture the output logs
    captured = capsys.readouterr()

    # Assert that the expected fallback message is present in the output logs
    expected_message = "Bruteforce method used instead"
    assert expected_message in captured.out, f"Expected '{expected_message}' message when falling back to brute-force."

# Test if query handles empty result set after applying filters
def test_query_empty_after_filters(setup_db):
    filters = [("metadata", {"info.type": "non_existent_type"})]
    results = setup_db.query("Abra", filters=filters)
    assert len(results) == 0, "Query with filters that result in an empty set should return an empty list."

# Test the correct index mapping of chunked documents when querying the database
def test_index_mapping_for_chunked_document():
    setup_db = HyperDB()

    # Add three documents: one below chunk size, one above (with specific word in second chunk), and another below chunk size
    doc1 = {"text": "word " * 100}  # Below chunk size
    doc2 = {"text": "word " * 505 + " uniqueword " + "word " * 100}  # Above chunk size, 'uniqueword' in second chunk
    doc3 = {"text": "word " * 200}  # Below chunk size

    setup_db.add(doc1)
    setup_db.add(doc2)
    setup_db.add(doc3)
    #setup_db.add([doc1, doc2, doc3])

    # Query the word that's specific to the second chunk of the second document
    query_input = "uniqueword"
    results = setup_db._execute_query(query_input, top_k=1, filters=[('sentence', 'uniqueword')], return_similarities=True)

    if not results:
        raise AssertionError("Query did not return any results")

    # Extract the returned document's index
    print(f"Source index: {setup_db.source_indices}")
    _, _, returned_index = results[0]

    # The expected original index for the second document (index 1, as it's the second document added)
    expected_original_index = 1

    # Check if source_indices maps back to the original document's index
    actual_original_index = setup_db.source_indices[returned_index]
    assert actual_original_index == expected_original_index, f"Incorrect index mapping for chunked document. Expected {expected_original_index}, got {actual_original_index}"

### Test caching functionality
def test_cache_miss_and_hit(setup_db):
    query_input = "Abra"  # Example query input
    
    # First query to fill the cache
    setup_db.query(query_input)
    cache_info1 = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info1['hits'] == 0, f"Expected 0 cache hits, but got {cache_info1['hits']}"
    assert cache_info1['misses'] == 1, f"Expected 1 cache miss, but got {cache_info1['misses']}"
    
    # Second query with the same input to hit the cache
    setup_db.query(query_input)
    cache_info2 = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info2['hits'] == 1, f"Expected 1 cache hit, but got {cache_info2['hits']}"
    assert cache_info2['misses'] == 1, f"Expected 1 cache miss, but got {cache_info2['misses']}"

def test_change_cache_size(setup_db):
    maxsize = 128  # Example max size
    setup_db.lru_cache = cachetools.LRUCache(maxsize=maxsize)  # Update cache size
    cache_info3 = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info3['maxsize'] == maxsize, f"Expected maxsize to be {maxsize}, but got {cache_info3['maxsize']}"

def test_cache_eviction(setup_db):
    maxsize = 2
    setup_db.lru_cache = cachetools.LRUCache(maxsize=maxsize)  # Update cache size
    for i in range(maxsize + 1):  # One more than max size to trigger eviction
        query_input = f"Query {i}"
        setup_db.query(query_input)
    cache_info4 = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info4['currsize'] == maxsize, f"Expected currsize to be {maxsize}, but got {cache_info4['currsize']}"

def test_cache_clearing_on_add_remove_document():
    setup_db = HyperDB()
    # Add a document then query to fill the cache
    setup_db.add({"text": "Sample document"})
    setup_db.query("Sample query")  # This should fill the cache
    cache_info_before = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info_before['currsize'] > 0, "Cache should be filled before clearing"

    # Add a new document and check if cache is cleared
    setup_db.add({"text": "Another document"})
    cache_info_after_add = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info_after_add['currsize'] == 0, "Cache should be cleared after adding a document"

    # Make another query to fill the cache again
    setup_db.query("Another query")
    cache_info_before_remove = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info_before_remove['currsize'] > 0, "Cache should be filled before clearing"

    # Remove a document and check if cache is cleared
    setup_db.remove_document(0)  # Assuming this index exists
    cache_info_after_remove = setup_db.get_cache_size_and_info()['cache_info']
    assert cache_info_after_remove['currsize'] == 0, "Cache should be cleared after removing a document"

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
