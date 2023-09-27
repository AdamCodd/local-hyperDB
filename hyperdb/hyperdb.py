import gzip
import pickle
import json
import sqlite3
import datetime
import numpy as np
import collections
import string
import torch
import onnxruntime as ort
from contextlib import closing
from transformers import BertTokenizerFast
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

import hyperdb.ranking_algorithm as ranking

ort.set_default_logger_severity(3) # Disable onnxruntime useless warnings when switching to GPU
EMBEDDING_MODEL = None
tokenizer = None
MAX_LENGTH = 510 # 512 - 2 to account for special tokens added by the BERT tokenizer

# sentence-transformers/all-MiniLM-L6-v2 use 384 dimensional vectors 
# sentence-transformers/all-mpnet-base-v2 (onnx format) use 768 dimensional vectors (match all transformers model)

def initialize_model():
    global EMBEDDING_MODEL, tokenizer
    if EMBEDDING_MODEL is not None and tokenizer is not None:
        return
    else:
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
        tokenizer = BertTokenizerFast.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')

def text_to_chunks(text, tokenizer, max_length=MAX_LENGTH):
    encoding = tokenizer([text], truncation=False, padding='longest', return_tensors='pt')
    input_ids = encoding['input_ids'][0]
    
    chunks = []
    for i in range(0, len(input_ids), max_length):
        chunk_tokens = input_ids[i:i + max_length]
        chunk = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk)
    return chunks

def prepare_texts_and_indices(documents):
    texts = []
    source_indices = []
    split_info = {}
    
    if documents is None or not documents:
        raise ValueError("Documents cannot be empty or None.")

    def process_text(text, index):
        nonlocal texts, source_indices, split_info
        chunks = text_to_chunks(text, tokenizer)
        texts.extend(chunks)
        source_indices.extend([index] * len(chunks))
        if len(chunks) > 1:
            split_info[index] = len(chunks)  # Store the number of chunks

    if isinstance(documents, str):
        process_text(documents, 0)
        return texts, source_indices, split_info

    elif isinstance(documents, list):
        for i, doc in enumerate(documents):
            if isinstance(doc, dict):
                # Concatenate all values to form a single string for each document
                doc_text = " ".join(str(val) for val in doc.values())
                process_text(doc_text, i)
            elif isinstance(doc, list):  # Handling nested lists
                for sub_doc in doc:
                    process_text(str(sub_doc), i)
            elif isinstance(doc, str):
                process_text(doc, i)
            else:
                raise ValueError("Unsupported document type.")
    else:
        raise ValueError("Documents should either be a string or a list.")

    return texts, source_indices, split_info


def get_embedding(documents, fp_precision=np.float32):
    initialize_model()

    if documents is None:
        raise ValueError("Documents cannot be None.")
    
    try:
        texts, source_indices, split_info = prepare_texts_and_indices(documents)
        embeddings = EMBEDDING_MODEL.encode(texts).astype(fp_precision)
        embeddings = np.array(embeddings)
    except Exception as e:
        raise RuntimeError(f"An error occurred while generating embeddings: {e}")

    return embeddings, source_indices, split_info


class HyperDB:
    def __init__(
        self,
        documents=None,
        vectors=None,
        key=None,
        embedding_function=None,
        similarity_metric="cosine_similarity",
        fp_precision="float32",
        add_timestamp=False
    ):
        # Validate similarity metric upfront
        valid_metrics = [
            'dot_product', 'cosine_similarity', 'euclidean_metric', 'manhattan_distance',
            'jaccard_similarity', 'pearson_correlation', 'mahalanobis_distance', 'hamming_distance'
        ]
        if similarity_metric not in valid_metrics:
            raise ValueError("Unsupported similarity metric.")

        # Validate floating-point precision
        if fp_precision not in ["float16", "float32", "float64"]:
            raise ValueError("Unsupported floating-point precision.")
        
        self.source_indices = []
        self.split_info = {}
        self.documents = []
        self.vectors = None
        self.key = key
        self.add_timestamp = add_timestamp
        self.compiled_keys = None
        self.fp_precision = getattr(np, fp_precision)
        self.embedding_function = embedding_function or (
            lambda docs: get_embedding(docs, fp_precision=fp_precision)
        )
        self.similarity_metric = similarity_metric

        # Initialize temporary storage for new vectors, documents, and source indices
        self.pending_vectors = []
        self.pending_documents = []
        self.pending_source_indices = []
        
        # If vectors are provided, use them; otherwise, add documents using add_documents
        if vectors is not None:
            self.vectors = vectors
            self.documents = documents
        elif documents:
            self.add(documents, vectors=None, add_timestamp=self.add_timestamp)
        
        # Compile keys if needed
        if self.key:
            self.compile_keys()
            
        # Store vector query
        self.query_vector_cache = {}


    def commit_pending(self):
        """Commit the pending vectors and documents to the main storage."""
        # Pre-checks and Initialization
        if len(self.pending_vectors) == 0:
            return
        
        # Calculate total number of new vectors
        total_new_vectors = sum([vec.shape[0] for vec in self.pending_vectors])
        
        # Consistency check between pending vectors and pending documents
        if total_new_vectors != len(self.pending_documents):
            print(f"Inconsistency detected between the number of pending vectors and documents. "
                  f"Total vectors calculated: {total_new_vectors}, Total pending documents: {len(self.pending_documents)}. "
                  "Transaction rolled back.")
            return
        
        # Initialize or resize self.vectors
        next_index = 0
        if self.vectors is None:
            self.vectors = np.zeros((total_new_vectors, self.pending_vectors[0].shape[1]), dtype=self.fp_precision)
        else:
            next_index = self.vectors.shape[0]
            self.vectors = np.resize(self.vectors, (self.vectors.shape[0] + total_new_vectors, self.vectors.shape[1]))

        # Transactional Commit
        try:
            # Add Pending Vectors
            for vec in self.pending_vectors:
                end_index = next_index + vec.shape[0]
                self.vectors[next_index:end_index, :] = vec
                next_index = end_index
            
            # Add Pending Source Indices
            new_source_indices = []  # Temporary list to store new source indices
            source_index_offset = len(self.documents)  # Offset for source indices in the main list
            for source_index in self.pending_source_indices:
                if source_index in self.split_info:
                    count = self.split_info[source_index]
                    new_source_indices.extend([source_index_offset + source_index] * count)
                else:
                    new_source_indices.append(source_index_offset + source_index)
            
            # Consistency check
            if len(new_source_indices) != total_new_vectors:
                raise ValueError("Inconsistency detected in new source indices.")
                       
            # Commit to main storage
            self.source_indices.extend(new_source_indices)
            self.documents.extend(self.pending_documents)
            
        except Exception as e:
            # Rollback transaction
            print(f"Error occurred during commit: {e}. Rolling back transaction.")
            self.vectors = self.vectors[:next_index, :]  # Remove the newly added vectors
            return

        # Cleanup
        self.pending_vectors.clear()
        self.pending_documents.clear()
        self.pending_source_indices.clear()


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
        if not documents:
            return
        if isinstance(documents, list):
            self.add_documents(documents, vectors, add_timestamp)
        else:
            self.add_document(documents, vectors, add_timestamp=add_timestamp)
            self.commit_pending()

    def compile_keys(self):
        self.compiled_keys = []
        if isinstance(self.key, str):  # Handle single string case
            self.compiled_keys.append([self.key])
        elif isinstance(self.key, list):  # Handle list of strings case
            for k in self.key:
                self.compiled_keys.append(k.split('.'))

    def filter_document(self, document):
        """Filters a document based on a key."""
        if not self.compiled_keys or not isinstance(document, dict):
            return document

        try:
            filtered_doc = {}
            for nested_keys in self.compiled_keys:
                value = self.get_nested_value(document, nested_keys)
                if value is not None:
                    sub_key = nested_keys[-1]
                    filtered_doc[sub_key] = value
        except Exception as e:
            raise ValueError(f"Error in filtering document by key: {e}")

        return filtered_doc if filtered_doc else document

    def add_document(self, document, vectors=None, count=1, add_timestamp=False):
        """
        Add a single document to the database.
        Args:
            document: The document to add. Could be of any type.
            vectors (list): Pre-computed vector for the document.
            count (int): Number of times to add the document.
            add_timestamp (bool): Whether to add a timestamp to the document if it's a dictionary. Default is False.
        """
        
        if not document:
            return
        
        document = self.filter_document(document)
        split_info = {}
        
        # Only add a timestamp if the document is a dictionary and add_timestamp is True
        if isinstance(document, dict) and add_timestamp:
            timestamp = datetime.datetime.now().timestamp()
            document['timestamp'] = float(timestamp)  # Store timestamp as float
        
        # Temporary copies of pending lists to maintain transactional integrity
        temp_pending_vectors = self.pending_vectors.copy()
        temp_pending_documents = self.pending_documents.copy()
        temp_pending_source_indices = self.pending_source_indices.copy()
        
        if vectors is None:
            vectors, source_indices, split_info = self.embedding_function([document])
            
            # Handle split_info
            for i, chunk_count in split_info.items():
                index = len(self.documents) + len(temp_pending_documents) + i
                self.split_info[index] = chunk_count
        
        if vectors.size == 0:  # Check if vectors is empty
            raise ValueError("No vectors returned by the embedding_function.")
            
        if len(vectors.shape) == 1:
            vectors = np.expand_dims(vectors, axis=0)
            
        if len(vectors.shape) != 2:
            raise ValueError("Vectors does not have the expected structure.")
       
        # Append to pending vectors
        temp_pending_vectors.append(vectors)
        
        # Append to pending documents and source indices
        last_index = len(self.documents) + len(temp_pending_documents)
        for _ in range(count):
            temp_pending_documents.append(document)
            
            if split_info:  # If the document has been chunked
                for i, chunk_count in split_info.items():
                    temp_pending_source_indices.extend([last_index] * chunk_count)
            else:
                temp_pending_source_indices.append(last_index)
            
            last_index += 1  # Increment the last index for the next iteration
        
        # Commit
        self.pending_vectors = temp_pending_vectors
        self.pending_documents = temp_pending_documents
        self.pending_source_indices = temp_pending_source_indices



    def add_documents(self, documents, vectors=None, add_timestamp=False):
        """
        Add multiple documents to the database in a transactional manner.
        Args:
            documents (list): A list of documents to add.
            vectors (list): Pre-computed vectors for the documents. Should match the length and order of documents if provided.
            add_timestamp (bool): Whether to add a timestamp to the documents if they are dictionaries. Default is False.
        """

        
        # Preliminary Checks
        if not documents:  # Check for empty documents
            return
        
        if vectors is not None and len(documents) != len(vectors):
            print("Error: The number of documents must match the number of vectors.")
            return

        try:
            # Data preparation
            filtered_documents = [self.filter_document(doc) for doc in documents]
            
            if vectors is None:
                vectors, source_indices, split_info = self.embedding_function(filtered_documents)
            else:
                source_indices = list(range(len(documents)))

            if vectors.size == 0:  # Check for empty vectors
                raise ValueError("No vectors returned by the embedding_function.")
            
            # Add to Pending Lists
            temp_pending_vectors = self.pending_vectors.copy()
            temp_pending_source_indices = self.pending_source_indices.copy()
            temp_pending_documents = self.pending_documents.copy()

            temp_pending_vectors.append(vectors)
            temp_pending_source_indices.extend(source_indices)

            for i, document in enumerate(documents):           
                count = split_info.get(i, 1)
                self.add_document(document, vectors[i], count, add_timestamp)

            # Calculate total number of vectors in temp_pending_vectors
            total_vectors = sum(vec.shape[0] for vec in temp_pending_vectors)

            # Commit to Main Storage
            if total_vectors == len(self.pending_documents):
                self.pending_vectors = temp_pending_vectors
                self.pending_source_indices = temp_pending_source_indices
                self.commit_pending()
            else:
                print(f"Inconsistency in add_documents detected between the number of pending vectors and documents. "
          f"Total vectors calculated: {total_vectors}, Total pending documents: {len(self.pending_documents)}. "
          "Transaction rolled back.")
                self.pending_documents = temp_pending_documents  # Rollback

        except Exception as e:
            print(f"An exception occurred: {e}")
            self.pending_documents = temp_pending_documents  # Rollback in case of exceptions

  
    def remove_document(self, indices):
        """
        Remove documents from the database by their indices.
        
        Args:
            indices (list or int): The index or list of indices of documents to remove.
        """
        # Ensure indices is a list
        if isinstance(indices, int):
            indices = [indices]

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
        
        # Create a set of all indices to be removed from source_indices
        indices_to_remove = set(indices)
        for idx in indices:
            if idx in self.split_info:
                num_chunks = self.split_info[idx]
                # Add all the chunk indices that belong to the same original document
                indices_to_remove.update(range(idx, idx + num_chunks))
        
        # Remove from source_indices
        self.source_indices = [idx for idx in self.source_indices if idx not in indices_to_remove]
        
        # Optionally, update split_info if needed
        if hasattr(self, 'split_info'):
            for idx in sorted(indices_to_remove, reverse=True):
                if idx in self.split_info:
                    del self.split_info[idx]


    def save(self, storage_file, format='pickle'):        
        # Check if there's nothing to save
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            print("Nothing to save. Exit.")
            return
        data = {
            "vectors": [vector.tolist() for vector in self.vectors],
            "documents": self.documents,
            "source_indices": self.source_indices,
            "split_info": self.split_info
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
                # Existing tables for documents and vectors
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
                
                # New tables for source_indices and split_info
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS source_indices (
                    id INTEGER PRIMARY KEY,
                    value INTEGER
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS split_info (
                    id INTEGER PRIMARY KEY,
                    value TEXT
                )
                ''')
                
                # Insert data into documents and vectors (existing behavior)
                for doc, vec in zip(data["documents"], data["vectors"]):
                    cursor.execute('INSERT INTO documents (data) VALUES (?)', (json.dumps(doc),))
                    doc_id = cursor.lastrowid
                    cursor.execute('INSERT INTO vectors (document_id, vector) VALUES (?, ?)', (doc_id, json.dumps(vec)))
                
                # Insert data into source_indices and split_info (new behavior)
                for index in data["source_indices"]:
                    cursor.execute('INSERT INTO source_indices (value) VALUES (?)', (index,))
                    
                cursor.execute('INSERT INTO split_info (value) VALUES (?)', (json.dumps(data["split_info"]),))
                
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
            self.source_indices = data.get("source_indices", [])
            self.split_info = data.get("split_info", {})
        except Exception as e:
            print(f"An exception occurred during load: {e}")

    def _load_pickle(self, storage_file):
        try:
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        except OSError:
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
        source_indices = []
        split_info = {}

        try:
            # Existing data load for documents and vectors
            for row in cursor.execute('SELECT data FROM documents'):
                documents.append(json.loads(row[0]))

            for row in cursor.execute('SELECT vector FROM vectors ORDER BY document_id'):
                vectors.append(json.loads(row[0]))

            # New data load for source_indices and split_info
            for row in cursor.execute('SELECT value FROM source_indices'):
                source_indices.append(row[0])

            for row in cursor.execute('SELECT value FROM split_info'):
                split_info = json.loads(row[0])

        except sqlite3.Error as e:
            print(f"SQLite error: {e}")

        conn.close()
        return {"vectors": vectors, "documents": documents, "source_indices": source_indices, "split_info": split_info}
        
    def compute_and_save_word_frequencies(self, output_file_path):
        """
        Compute word frequencies from the documents in the database and save them to a text file.
        """
        word_frequencies = collections.defaultdict(int)
        
        # Compute word frequencies
        for document in self.documents:
            if isinstance(document, dict):
                for key, value in document.items():
                    cleaned_value = str(value).translate(str.maketrans('', '', string.punctuation))
                    words = cleaned_value.split()
                    for word in words:
                        word_frequencies[word.lower()] += 1
            elif isinstance(document, str):
                cleaned_value = document.translate(str.maketrans('', '', string.punctuation))
                words = cleaned_value.split()
                for word in words:
                    word_frequencies[word.lower()] += 1
        
        # Sort by frequency
        sorted_word_frequencies = sorted(word_frequencies.items(), key=lambda x: x[1], reverse=True)
        
        # Save to text file
        with open(output_file_path, 'w') as f:
            for word, freq in sorted_word_frequencies:
                f.write(f"{word}: {freq}\n")    
        

    def get_nested_value(self, dictionary, keys):
        """     
        Retrieves a nested value from a dictionary by following a sequence of keys.
        """
        try:
            value = dictionary
            for key in keys:
                if isinstance(value, list):
                    value = [sub_value.get(key, None) for sub_value in value if isinstance(sub_value, dict)]
                else:
                    value = value.get(key, None)
            return value
        except (KeyError, TypeError, AttributeError):
            return None

    def filter_by_key(self, vectors, documents, keys):
        if not isinstance(keys, list):
            keys = [keys]

        filtered_vectors_dict = {}
        filtered_documents_dict = {}

        for key in keys:
            nested_keys = key.split('.') if '.' in key else [key]
            for vec, doc in zip(vectors, documents):
                doc_id = id(doc)
                sub_text = self.get_nested_value(doc, nested_keys)

                if isinstance(sub_text, list):
                    for item in sub_text:
                        if item is not None:
                            new_vec = self.embedding_function([str(item)])[0]
                            if doc_id in filtered_vectors_dict:
                                filtered_vectors_dict[doc_id].append(new_vec)
                            else:
                                filtered_vectors_dict[doc_id] = [new_vec]
                                filtered_documents_dict[doc_id] = doc
                elif sub_text is not None:
                    new_vec = self.embedding_function([str(sub_text)])[0]
                    if doc_id not in filtered_vectors_dict:
                        filtered_vectors_dict[doc_id] = [new_vec]
                        filtered_documents_dict[doc_id] = doc
                    else:
                        filtered_vectors_dict[doc_id].append(new_vec)

        # Average the vectors for each document
        for doc_id, vec_list in filtered_vectors_dict.items():
            last_dim_size = np.array(vec_list).shape[-1]
            reshaped_vec_list = np.array(vec_list).reshape(-1, last_dim_size)
            filtered_vectors_dict[doc_id] = np.mean(reshaped_vec_list, axis=0)

        filtered_vectors = np.array(list(filtered_vectors_dict.values()))
        filtered_documents = list(filtered_documents_dict.values())

        last_dim_size = filtered_vectors.shape[-1]
        filtered_vectors = filtered_vectors.reshape(-1, last_dim_size)

        return filtered_vectors, filtered_documents


    def generate_query_vector(self, query_text):
        query_vector = self.embedding_function([query_text])
        if not query_vector:
            raise ValueError("Failed to generate an embedding for the query text.")
        return query_vector[0]

    def filter_vectors_by_key(self, key):
        filtered_vectors, filtered_documents = self.filter_by_key(self.vectors, self.documents, key)
        if len(filtered_vectors) == 0:
            print(f"Warning: No documents were filtered using the key '{key}'. It might be non-existent or have null values.")
        return filtered_vectors, filtered_documents

    def apply_skip_doc(self, vectors, documents, skip_doc):
        """
        Skips a certain number of documents based on the skip_doc parameter.
        """
        if abs(skip_doc) > len(documents):
            print(f"Warning: The absolute value of skip_doc ({abs(skip_doc)}) is greater than the total number of documents ({len(documents)}).")
        if skip_doc > 0:
            return vectors[skip_doc:], documents[skip_doc:]
        elif skip_doc < 0:
            return vectors[:skip_doc], documents[:skip_doc]
        return vectors, documents

    def tokenize_sentence(self, sentence):
        """
        Tokenizes a sentence into words, removing punctuation.
        """
        tokens = sentence.lower().split()
        tokens = [''.join(c for c in t if c not in string.punctuation) for t in tokens]
        return tokens

    def filter_by_sentence(self, vectors, documents, sentence_filter):
        filtered_vectors = []
        filtered_documents = []
        sentence_filter = sentence_filter.lower()
        for vec, doc in zip(vectors, documents):
            tokens = self.tokenize_sentence(str(doc))
            if sentence_filter in tokens:
                filtered_vectors.append(vec)
                filtered_documents.append(doc)
        return filtered_vectors, filtered_documents


    def _generate_and_validate_query_vector(self, query_input):
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            raise ValueError("The database is empty. Cannot proceed with the query.")

        if isinstance(query_input, str):
            if query_input in self.query_vector_cache:
                return self.query_vector_cache[query_input]
            else:
                query_vector = self.generate_query_vector(query_input)
                self.query_vector_cache[query_input] = query_vector
        elif isinstance(query_input, (list, np.ndarray)):
            query_input = np.array(query_input)  # Convert to NumPy array for uniformity

            # Check type of elements
            if query_input.dtype.type not in (np.int_, np.float_, np.intc, np.intp, np.int8,
                                              np.int16, np.int32, np.int64, np.uint8,
                                              np.uint16, np.uint32, np.uint64,
                                              np.float16, np.float32, np.float64):
                raise ValueError("Numeric array-like query_input expected.")

            # Check dimensionality
            if query_input.ndim > 2:
                raise ValueError("query_input must be a 1D or 2D array.")

            # If 1D, convert to 2D
            if query_input.ndim == 1:
                query_input = np.array([query_input])

            # Validate dimensions
            if query_input.shape[1] != self.vectors.shape[1]:
                raise ValueError(f"The dimension of the query_vector ({query_input.shape[1]}) must match the dimension of the vectors in the database ({self.vectors.shape[1]}).")
            
            query_vector = query_input  # Already a NumPy array, no need for further conversion
        else:
            raise ValueError("query_input must be either a string or a numeric array-like object.")

        if query_vector.size == 0:
            raise ValueError("The generated query vector is empty.")
        
        return query_vector


    def query(self, query_input, top_k=5, return_similarities=True, key=None, recency_bias=0, timestamp_key=None, skip_doc=0, sentence_filter=None, metric='cosine_similarity'):  
        """
        Query the document store to retrieve relevant documents based on a variety of optional parameters.
        
        Parameters:
        - query_input (str or array-like): The query as a string or as a vector.
        - top_k (int): The number of top matches to return.
        - return_similarities (bool): Whether to return similarity scores along with documents.
        - key (str): A key to filter the documents by.
        - recency_bias (float): A factor to bias toward more recent documents.
        - timestamp_key (str): The key to use for timestamps in the documents.
        - skip_doc (int): The number of documents to skip.
        - sentence_filter (str): A sentence to filter the documents by.
        - metric (str): Optional. Override the default similarity metric for this query.
        """    
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            raise ValueError("The database is empty. Cannot proceed with the query.")
  
        # Decide which similarity metric to use for this query
        effective_metric = metric if metric is not None else self.similarity_metric
        
        try:
            query_vector = self._generate_and_validate_query_vector(query_input)    
            
            filtered_vectors = self.vectors
            filtered_documents = self.documents
            if len(filtered_vectors) != len(filtered_documents):
                print(f"Inconsistency detected between filtered vectors {len(filtered_vectors)} and filtered documents {len(filtered_documents)}.")
                return []
                
            # 1. Apply key-based filter if provided
            if key:
                filtered_vectors, filtered_documents = self.filter_vectors_by_key(key)
            # 2. Apply sentence_filter if provided
            if sentence_filter:
                filtered_vectors, filtered_documents = self.filter_by_sentence(filtered_vectors, filtered_documents, sentence_filter)

            # 3. Apply skip-doc filter if provided
            if skip_doc != 0:
                filtered_vectors, filtered_documents = self.apply_skip_doc(filtered_vectors, filtered_documents, skip_doc)

            # Convert to NumPy array for computation
            filtered_vectors = np.array(filtered_vectors, dtype=self.fp_precision)

            # Check if filtered_vectors is empty
            if filtered_vectors.size == 0:
                print("No document matches your query.")
                return []

            # Decide which ranking algorithm to use based on timestamp_key
            if timestamp_key:
                timestamps = [document.get(timestamp_key, 0) for document in filtered_documents]
                timestamps = np.array(timestamps, dtype=float)  # Convert to float if not in float already
            else:
                timestamps = None

            ranked_results, scores = ranking.hyperDB_ranking_algorithm_sort(
                filtered_vectors, query_vector, top_k=top_k, metric=metric, timestamps=timestamps, recency_bias=recency_bias
            )
                    
            # Debugging query
            # print("Debugging Information:")
            # print(f"Source Indices: {self.source_indices}")
            # print(f"Split Info: {self.split_info}")
            # print(f"Ranked Results: {ranked_results}")
                    
            # Validate ranked_results
            if max(ranked_results) >= len(filtered_documents):
                raise IndexError(f"Invalid index in ranked_results. Max index: {max(ranked_results)}, Length of filtered_documents: {len(filtered_documents)}")

            if return_similarities:
                return list(zip([filtered_documents[index] for index in ranked_results], scores))
            
            return [filtered_documents[index] for index in ranked_results]

        except (ValueError, TypeError) as e:
            print(f"An exception occurred due to invalid input: {e}")
            return []
        except Exception as e:
            print(f"An unknown exception occurred: {e}")
            return []
