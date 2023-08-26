import gzip
import pickle
import json
import sqlite3
import datetime
import numpy as np
import collections
import string
from contextlib import closing
from transformers import BertTokenizer
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

from hyperdb.ranking_algorithm import (
    dot_product,
    adams_similarity,
    cosine_similarity,
    derridaean_similarity,
    euclidean_metric,
    hyper_SVM_ranking_algorithm_sort,
    custom_ranking_algorithm_sort
)

EMBEDDING_MODEL = None
tokenizer = None

#Maximum max_length=256 for all-MiniLM-L6-v2
def text_to_chunks(text, tokenizer, max_length=256):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []

    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i+max_length]
        chunk = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk)

    return chunks

def get_embedding(documents, key=None, fp_precision=np.float32):
    """Embedding function that uses Sentence Transformers."""
    global EMBEDDING_MODEL, tokenizer
    if EMBEDDING_MODEL is None or tokenizer is None:
        # Initialize the model and tokenizer
        EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

    texts = []
    source_indices = []
    split_info = {}

    # Check if documents is a string or list
    if isinstance(documents, str):
        chunks = text_to_chunks(documents, tokenizer)
        if len(chunks) > 1:
            split_info[0] = True
        for chunk in chunks:
            texts.append(chunk)
            source_indices.append(0)
    elif isinstance(documents, list):
        if not documents:
            raise ValueError("The document list is empty.")
        
        # Nested dictionary keys
        if isinstance(documents[0], dict):
            for i, doc in enumerate(documents):
                if not isinstance(doc, dict):
                    raise ValueError(f"Expected a dictionary at index {i}, but got {type(doc)}.")

                if isinstance(key, str):
                    nested_key_chain = key.split(".") if "." in key else [key]
                    text = doc
                    for nested_key in nested_key_chain:
                        try:
                            text = text[nested_key]
                        except KeyError:
                            raise KeyError(f"Key '{nested_key}' not found in the document at index {i}.")
                    
                    if not isinstance(text, str):
                        raise ValueError(f"Expected a string for key '{key}', but got {type(text)}.")
                    
                    chunks = text_to_chunks(text, tokenizer)
                    if len(chunks) > 1:
                        split_info[i] = True
                    for chunk in chunks:
                        texts.append(f"{nested_key}: {chunk}")
                        source_indices.append(i)
                elif key is None:
                    for k, value in doc.items():
                        if not isinstance(value, str):
                            raise ValueError(f"Expected a string for key '{k}', but got {type(value)}.")
                        
                        chunks = text_to_chunks(value, tokenizer)
                        if len(chunks) > 1:
                            split_info[i] = True
                        for chunk in chunks:
                            texts.append(f"{k}: {chunk}")
                            source_indices.append(i)
        elif isinstance(documents[0], str):
            texts = documents
            source_indices = list(range(len(documents)))
        else:
            raise ValueError("Unsupported document type.")
    else:
        raise ValueError("Documents should either be a string or a list.")
    embeddings = EMBEDDING_MODEL.encode(texts).astype(fp_precision)
    
    return embeddings, source_indices, split_info



class HyperDB:
    def __init__(
        self,
        documents=None,
        vectors=None,
        key=None,
        embedding_function=None,
        similarity_metric="cosine",
        use_word_frequencies=False,
        fp_precision="float32"  # Set floating-point precision - Default: float32
    ):  
        """
        Initialize the HyperDB instance.

        Args:
            documents (list): A list of documents to initialize the database with.
            vectors (list): A list of pre-computed vectors. If provided, it should match the length and order of documents.
            key (str): The key to extract text from the documents when they are dictionaries.
            embedding_function (callable): A function to compute document embeddings. Default is None.
            similarity_metric (str): The metric used to compute similarities ('dot', 'cosine', 'euclidean', 'adams', or 'derrida').
        """
        
        self.source_indices = []
        self.split_info = {}
        documents = documents or []
        self.documents = []
        self.pending_vectors = []  # Store vectors that need to be stacked
        self.vectors = None
        
        self.fp_precision = getattr(np, fp_precision)   # Convert the string to a NumPy dtype
        
        self.embedding_function = embedding_function or (
            lambda docs: get_embedding(docs, key=key, fp_precision=fp_precision)
        )
        
        self.use_word_frequencies = use_word_frequencies
        if self.use_word_frequencies:
            self.word_frequencies = collections.defaultdict(int) # Initialize word frequencies attribute
            
        if vectors is not None:
            self.vectors = vectors
            self.documents = documents
        else:
            self.add_documents(documents)

        if similarity_metric.__contains__("dot"):
            self.similarity_metric = dot_product
        elif similarity_metric.__contains__("cosine"):
            self.similarity_metric = cosine_similarity
        elif similarity_metric.__contains__("euclidean"):
            self.similarity_metric = euclidean_metric
        elif similarity_metric.__contains__("derrida"):
            self.similarity_metric = derridaean_similarity
        elif similarity_metric.__contains__("adams"):
            self.similarity_metric = adams_similarity
        else:
            raise Exception("Similarity metric not supported. Please use either 'dot', 'cosine', 'euclidean', 'adams', or 'derrida'.")

    def _get_word_freq_db_path(self, include_timestamp=False):
        """Helper method to determine the path for the word frequencies database."""
        if hasattr(self, 'storage_file'):
            timestamp = ""
            if include_timestamp:
                current_timestamp = datetime.datetime.now().strftime('%d%m%Y%H%M%S')
                timestamp = f"_{current_timestamp}"
                
            # Handle the pickle format
            if self.storage_file.endswith('.pickle.gz'):
                return self.storage_file.replace('.pickle.gz', f'_word_freqs{timestamp}.pickle.gz')
            # Handle the json format
            elif self.storage_file.endswith('.json'):
                return self.storage_file.replace('.json', f'_word_freqs{timestamp}.json')
            # Handle the sqlite format
            elif self.storage_file.endswith('.sqlite'):
                return self.storage_file.replace('.sqlite', f'_word_freqs{timestamp}.sqlite')
            else:
                # If the extension is not recognized, raise an error
                raise ValueError(f"Unrecognized file extension in {self.storage_file}.")
        else:
            raise ValueError("The main database storage path is not set.")


    def _process_document_for_word_frequencies(self, document):
        """Update word frequencies based on the given document."""
        # If the document is a dictionary, process its key-value pairs
        if isinstance(document, dict):
            for key, value in document.items():
                # Skip processing for the 'timestamp' key
                if key == 'timestamp':
                    continue
                # If value is a string, process its words
                if isinstance(value, str):  
                    cleaned_value = value.translate(str.maketrans('', '', string.punctuation))
                    words = cleaned_value.split()  # Split the value into words
                    for word in words:
                        self.word_frequencies[word.lower()] += 1
        # If document is a string, process its words
        elif isinstance(document, str):
            cleaned_value = document.translate(str.maketrans('', '', string.punctuation))
            words = cleaned_value.split()
            for word in words:
                self.word_frequencies[word.lower()] += 1
                
    def _initialize_word_frequencies(self):
        """Initializes the word frequencies from the current documents in the database."""
        if not self.use_word_frequencies:
            return
        for document in self.documents:
            self._process_document_for_word_frequencies(document)
            
    def recompute_word_frequencies(self):
        """
        Recompute word frequencies for the entire document database and save them.
        """
        self.word_frequencies = collections.defaultdict(int)  # Reset word frequencies
        self._initialize_word_frequencies()
        self.save_word_freqs()


    def get_word_frequencies(self):
        """
        Returns a dictionary of word frequencies across all documents in the database.
        """
        return self.word_frequencies
    

    def _determine_format_from_path(self, storage_file):
        """Determine the database format based on the file extension."""
        if storage_file.endswith('.pickle.gz'):
            return 'pickle'
        elif storage_file.endswith('.json'):
            return 'json'
        elif storage_file.endswith('.sqlite'):
            return 'sqlite'
        else:
            raise ValueError(f"Unrecognized file extension in {storage_file}.")

    def save_word_freqs(self, include_timestamp=False):
        """Save word frequencies to a separate database."""
        if not self.use_word_frequencies:
            return

        storage_file = self._get_word_freq_db_path(include_timestamp)
        format = self._determine_format_from_path(storage_file)

        # Save
        try:
            if format == 'pickle':
                self._save_pickle(storage_file, self.word_frequencies)
            elif format == 'json':
                self._save_json(storage_file, self.word_frequencies)
            elif format == 'sqlite':
                self._save_sqlite(storage_file, {"word_frequencies": self.word_frequencies})
            else:
                raise ValueError(f"Unsupported format '{format}'")
        except Exception as e:
            print(f"An exception occurred during word frequencies save: {e}")

    def load_word_freqs(self):
        """Load word frequencies from a separate database."""
        if not self.use_word_frequencies:
            return

        storage_file = self._get_word_freq_db_path()
        format = self._determine_format_from_path(storage_file)

        # Load
        try:
            if format == 'pickle':
                self.word_frequencies = self._load_pickle(storage_file)
            elif format == 'json':
                self.word_frequencies = self._load_json(storage_file)
            elif format == 'sqlite':
                data = self._load_sqlite(storage_file)
                if "word_frequencies" in data:
                    self.word_frequencies = data["word_frequencies"]
                else:
                    self.word_frequencies = collections.defaultdict(int)
            else:
                raise ValueError(f"Unsupported format '{format}'")
        except FileNotFoundError:
            self.word_frequencies = collections.defaultdict(int)
        except Exception as e:
            print(f"An exception occurred during word frequencies load: {e}")

    def size(self):
        """
        Returns the number of documents in the database.
        """
        return len(self.documents)

    def dict(self, vectors=False):
        """
        Returns the database in dictionary format.
        Args:
            vectors (bool): If True, include vectors in the output.
        Returns:
            list: List of dictionaries, each representing a document and its associated metadata.
        """
        if vectors:
            return [
                {"document": document, "vector": vector.tolist(), "index": index}
                for index, (document, vector) in enumerate(
                    zip(self.documents, self.vectors)
                )
            ]
        return [
            {"document": document, "index": index}
            for index, document in enumerate(self.documents)
        ]

    def add(self, documents, vectors=None, add_timestamp=False):
        """
        Add documents to the database.
        Args:
            documents (list or dict): A list of documents or a single document.
            vectors (list): Pre-computed vectors for the documents. If provided, should match the length and order of documents.
            add_timestamp (bool): Whether to add a timestamp to the documents if they are dictionaries. Default is False.
        """
        if not isinstance(documents, list):
            return self.add_document(documents, vectors, add_timestamp=add_timestamp)
        self.add_documents(documents, vectors, add_timestamp=add_timestamp)

    def add_document(self, document, vectors=None, count=1, update_word_freqs=True, add_timestamp=False):
        """
        Add a single document to the database.

        Args:
            document: The document to add. Could be of any type.
            vectors (list): Pre-computed vector for the document.
            count (int): Number of times to add the document.
            update_word_freqs (bool): Whether to update word frequencies. Default is True.
            add_timestamp (bool): Whether to add a timestamp to the document if it's a dictionary. Default is False.
        """
        if vectors is None:
            vectors, _, _ = self.embedding_function([document])

        if vectors.size == 0:  # Check if vectors is empty
            raise ValueError("No vectors returned by the embedding_function.")

        # Expand dimensions if vectors is 1D array (single document)
        if len(vectors.shape) == 1:
            vectors = np.expand_dims(vectors, axis=0)

        if len(vectors.shape) != 2:  # Raise an error if vectors does not have a 2D shape (should be num_documents x embedding_dim)
            raise ValueError("Vectors does not have the expected structure.")

        if self.vectors is None:
            self.vectors = np.empty((0, vectors.shape[1]), dtype=self.fp_precision)

        vector_list = []
        for vector in vectors:
            if len(vector) != self.vectors.shape[1]:
                raise ValueError(f"Dimension mismatch. Got vector of dimension {len(vector)} while the existing vectors are of dimension {self.vectors.shape[1]}.")

            vector_list.append(vector.astype(self.fp_precision))

        self.vectors = np.vstack([self.vectors, *vector_list])

        # Only add a timestamp if the document is a dictionary and add_timestamp is True
        if isinstance(document, dict) and add_timestamp:
            timestamp = datetime.datetime.now().timestamp()
            document['timestamp'] = str(timestamp)

        self.documents.extend([document]*count)  # Extend the document list with the same document for all chunks
        self.source_indices.extend([len(self.documents) - 1]*count)  # Extend the source_indices list with the same index for all chunks

        if self.use_word_frequencies and update_word_freqs:
            self._process_document_for_word_frequencies(document)


    def add_documents(self, documents, vectors=None, add_timestamp=False):
        """
        Add multiple documents to the database.

        Args:
            documents (list): A list of documents to add.
            vectors (list): Pre-computed vectors for the documents. If provided, should match the length and order of documents.
        """
        if not documents:
            return
        if vectors is None:
            vectors, source_indices, split_info = self.embedding_function(documents)
        else:
            source_indices = list(range(len(documents)))
        
        self.source_indices.extend(source_indices)  # store the source indices
        
        for i, document in enumerate(documents):
            count = split_info.get(i, 1)  # Get the number of chunks for this document
            self.add_document(document, vectors[i], count, update_word_freqs=False, add_timestamp=False)
            # Update word frequencies here:
            if self.use_word_frequencies:
                self._process_document_for_word_frequencies(document)


    def _decrement_word_frequencies(self, content):
        """Helper method to decrement word frequencies based on given content."""
        if isinstance(content, str):
            words = content.split()  # Split the content into words
            for word in words:
                word = word.lower()
                self.word_frequencies[word] -= 1
                if self.word_frequencies[word] == 0:
                    del self.word_frequencies[word]
        elif isinstance(content, dict):
            for key, value in content.items():
                # Skip processing for the 'timestamp' key
                if key == 'timestamp':
                    continue
                self._decrement_word_frequencies(value)
  
    def remove_document(self, indices):
        """
        Remove documents from the database by their indices.
        
        Args:
            indices (list or int): The index or list of indices of documents to remove.
        """
        # Ensure indices is a list
        if isinstance(indices, int):
            indices = [indices]

        # Before removing, decrement the word frequencies
        if self.use_word_frequencies:
            for idx in indices:
                document = self.documents[idx]
                self._decrement_word_frequencies(document)

        # Remove vectors
        if len(indices) == 1:
            # Efficiently exclude the index without recreating the array for single removal
            self.vectors = np.vstack([self.vectors[:indices[0]], self.vectors[indices[0]+1:]])
        else:
            # More efficient batch removal
            mask = np.ones(self.vectors.shape[0], dtype=bool)
            mask[indices] = False
            self.vectors = self.vectors[mask]

        # Reverse sort indices for safe batch popping from documents list
        for idx in sorted(indices, reverse=True):
            self.documents.pop(idx)
            # Optionally handle removal from source_indices if needed
            if idx in self.source_indices:
                self.source_indices.remove(idx)


    def save(self, storage_file, format='pickle'):
        # Check if there's nothing to save
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            return
        data = {
            "vectors": [vector.tolist() for vector in self.vectors],
            "documents": self.documents
        }
        
        try:
            if format == 'pickle':
                self._save_pickle(storage_file, data)
            elif format == 'json':
                self._save_json(storage_file, data)
            elif format == 'sqlite':
                self._save_sqlite(storage_file, data)
            else:
                raise ValueError(f"Unsupported format '{format}'")
        except Exception as e:
            print(f"An exception occurred during save: {e}")
    
    def _save_pickle(self, storage_file, data):
        try:
            if storage_file.endswith(".gz"):
                with gzip.open(storage_file, "wb") as f:
                    pickle.dump(data, f)
            else:
                with open(storage_file, "wb") as f:
                    pickle.dump(data, f)
        except Exception as e:
            print(f"An exception occurred during pickle save: {e}")

    def _save_json(self, storage_file, data):
        try:
            with open(storage_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"An exception occurred during JSON save: {e}")

    def _save_sqlite(self, storage_file, data):
        with closing(sqlite3.connect(storage_file)) as conn:
            cursor = conn.cursor()
            try:
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS documents (
                    id INTEGER PRIMARY KEY,
                    data TEXT
                )
                ''')

                cursor.execute('''
                CREATE TABLE IF NOT EXISTS vectors (
                    id INTEGER PRIMARY KEY,
                    document_id INTEGER,
                    vector BLOB
                )
                ''')

                for doc, vec in zip(data["documents"], data["vectors"]):
                    cursor.execute('INSERT INTO documents (data) VALUES (?)', (json.dumps(doc),))
                    doc_id = cursor.lastrowid
                    cursor.execute('INSERT INTO vectors (document_id, vector) VALUES (?, ?)', (doc_id, json.dumps(vec)))

                conn.commit()
            except sqlite3.Error as e:
                print(f"SQLite error: {e}")
                conn.rollback()

    def load(self, storage_file, format='pickle'):
        try:
            if format == 'pickle':
                data = self._load_pickle(storage_file)
            elif format == 'json':
                data = self._load_json(storage_file)
            elif format == 'sqlite':
                data = self._load_sqlite(storage_file)
            else:
                raise ValueError(f"Unsupported format '{format}'")

            self.vectors = np.array(data["vectors"], dtype=self.fp_precision)
            self.documents = data["documents"]
            self._initialize_word_frequencies()
        except Exception as e:
            print(f"An exception occurred during load: {e}")

    def _load_pickle(self, storage_file):
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        else:
            with open(storage_file, "rb") as f:
                data = pickle.load(f)
        return data

    def _load_json(self, storage_file):
        with open(storage_file, "r") as f:
            data = json.load(f)
        return data

    def _load_sqlite(self, storage_file):
        conn = sqlite3.connect(storage_file)
        cursor = conn.cursor()
        
        documents = []
        vectors = []
        
        for row in cursor.execute('SELECT data FROM documents'):
            documents.append(json.loads(row[0]))

        for row in cursor.execute('SELECT vector FROM vectors ORDER BY document_id'):
            vectors.append(json.loads(row[0]))
 
        conn.close()
        return {"vectors": vectors, "documents": documents}

    def query(self, query_text, top_k=5, return_similarities=True, recency_bias=0, use_timestamp=False, timestamp_key='timestamp'):
        # Check if there's nothing to query
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            return []
        try:
            query_vector = self.embedding_function([query_text])[0]
            # Decide which ranking algorithm to use based on the use_timestamp flag
            if use_timestamp:
                # Checking timestamps for each document
                timestamps = [document.get(timestamp_key, 0) for document in self.documents]  # Default to 0 if timestamp is not found
                ranked_results, combined_scores, original_similarities = custom_ranking_algorithm_sort(
                    self.vectors, query_vector, timestamps, top_k=top_k, metric=self.similarity_metric, recency_bias=recency_bias
                )
            else:
                # Use the hyper_SVM_ranking_algorithm_sort function
                ranked_results, original_similarities = hyper_SVM_ranking_algorithm_sort(
                    self.vectors, query_vector, top_k=top_k, metric=self.similarity_metric
                )
                combined_scores = original_similarities  # In this case, combined_scores are just the original similarities
              
            if return_similarities:
                return list(
                    zip([self.documents[index] for index in ranked_results], combined_scores, original_similarities)
                )
            return [self.documents[index] for index in ranked_results]
        except (ValueError, TypeError) as e:
            print(f"An exception occurred due to invalid input: {e}")
            return []
        except Exception as e:
            print(f"An unknown exception occurred: {e}")
            return []
