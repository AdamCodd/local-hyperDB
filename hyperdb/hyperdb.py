import gzip
import pickle
import json
import sqlite3
import datetime
import numpy as np
import collections
import string
import torch
import re
import onnxruntime as ort
from collections import Counter
from contextlib import closing
from transformers import BertTokenizerFast
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

import hyperdb.ranking_algorithm as ranking

ort.set_default_logger_severity(3) # Disable onnxruntime useless warnings when switching to GPU
EMBEDDING_MODEL = None
tokenizer = None
MAX_LENGTH = 510 # 512 - 2 to account for special tokens added by the BERT tokenizer

class HyperDB:
    """
    HyperDB is a class for efficient document storage and similarity search.
    
    Args:
        documents (list or None): List of documents to be stored in the database. 
        vectors (list or None): Precomputed vectors for the documents. 
        select_keys (list or None): List of the key(s) from the documents to include in the database.
        embedding_function (callable or None): Custom function to generate embeddings.
        fp_precision (str): Floating-point precision for embeddings.
            Supported values: 'float16', 'float32', 'float64'.
        add_timestamp (bool): Whether to add timestamps to documents.
        metadata_keys (list or None): Keys from the document to set as metadata for query-filtering later.
    """
    def __init__(
        self,
        documents=None,
        vectors=None,
        select_keys=None,
        embedding_function=None,
        fp_precision="float32",
        add_timestamp=False,
        metadata_keys=None,  # New parameter for metadata keys
    ):
        # Validate floating-point precision
        if fp_precision not in ["float16", "float32", "float64"]:
            raise ValueError("Unsupported floating-point precision.")
        
        self.source_indices = []
        self.split_info = {}
        self.documents = []
        self.vectors = None
        self.select_keys = select_keys
        self.add_timestamp = add_timestamp
        self.compiled_keys = None
        
        self.fp_precision = getattr(np, fp_precision)     
        self.initialize_model()
        self.embedding_function = embedding_function or self.get_embedding
        
        # Initialize temporary storage for new vectors, documents, and source indices
        self.pending_vectors = []
        self.pending_documents = []
        self.pending_source_indices = []
       
        # Compile keys if needed
        if self.select_keys:
            self.compile_keys()

        self._metadata_index = {}  # Key will be the unique doc index, value will be the metadata dictionary
        self.metadata_keys = metadata_keys or []  # Store the metadata keys  
  
        self.document_keys = []
  
        # Add 'timestamp' to metadata_keys if add_timestamp is True
        if self.add_timestamp and "timestamp" not in self.metadata_keys:
            self.metadata_keys.append("timestamp")
            self.document_keys.append("timestamp")
    
        # Assuming all documents have the same structure, we collect all unique keys
        if documents and isinstance(documents[0], dict):
           self.document_keys = self.collect_document_keys(documents)
           self.validate_keys(metadata_keys, self.document_keys, "metadata_keys", "document_keys")

        # If vectors are provided, use them; otherwise, add documents using add_documents
        if vectors is not None:
            self.vectors = vectors
            self.documents = documents
            self.source_indices = list(range(len(documents)))
        elif documents:
            self.add(documents, vectors=None, add_timestamp=self.add_timestamp)
            
        # Store vector query
        self.query_vector_cache = {}

    def initialize_model(self):
        """
        Initialize the embedding model.
        """
        global EMBEDDING_MODEL, tokenizer
        if EMBEDDING_MODEL is not None and tokenizer is not None:
            return

        model_choice = 'sentence-transformers/all-MiniLM-L6-v2'
        device = 'gpu' if torch.cuda.is_available() else 'cpu'
        EMBEDDING_MODEL = SentenceTransformer(model_choice, device=device)
        tokenizer = BertTokenizerFast.from_pretrained(model_choice)


    def text_to_chunks(self, text, max_length=MAX_LENGTH):
        """
        Splits a text into chunks of a given maximum length.
        
        Args:
            text (str): The text to be split.
            max_length (int): The maximum length for each chunk.
        """
        encoding = tokenizer([text], truncation=False, padding='longest', return_tensors='pt')
        input_ids = encoding['input_ids'][0]
        
        chunks = []
        for i in range(0, len(input_ids), max_length):
            chunk_tokens = input_ids[i:i + max_length]
            chunk = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
            chunks.append(chunk)
        return chunks

    def prepare_texts_and_indices(self, documents):
        """
        Prepares texts and their source indices.
        
        Args:
            documents (list or str): The documents to be prepared.
        """
        texts = []
        source_indices = []
        split_info = {}
        
        if documents is None or not documents:
            raise ValueError("Documents cannot be empty or None.")

        def process_text(text, index):
            nonlocal texts, source_indices, split_info
            chunks = self.text_to_chunks(text)
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
                    doc_text = " ".join(str(val) for val in doc.values())
                    process_text(doc_text, i)
                elif isinstance(doc, list):
                    for sub_doc in doc:
                        process_text(str(sub_doc), i)
                elif isinstance(doc, str):
                    process_text(doc, i)
                else:
                    raise ValueError("Unsupported document type.")
        else:
            raise ValueError("Documents should either be a string or a list.")

        return texts, source_indices, split_info

    def get_embedding(self, documents):
        """
        Generate embeddings for the given documents.
        
        Args:
            documents (list or str): The documents for which to generate embeddings.
        """
        if documents is None:
            raise ValueError("Documents cannot be None.")
        
        try:
            texts, source_indices, split_info = self.prepare_texts_and_indices(documents)
            if isinstance(EMBEDDING_MODEL, SentenceTransformer):
                embeddings = EMBEDDING_MODEL.encode(texts).astype(self.fp_precision)
            elif isinstance(EMBEDDING_MODEL, ort.InferenceSession):
                tokens = tokenizer(texts, padding=True, truncation=True, return_tensors="pt")
                ort_inputs = {'input_ids': tokens['input_ids'].numpy(), 'attention_mask': tokens['attention_mask'].numpy()}
                ort_outs = EMBEDDING_MODEL.run(None, ort_inputs)
                embeddings = ort_outs[0]
            else:
                raise ValueError("Unsupported model type.")
                
            embeddings = np.array(embeddings)
        except Exception as e:
            raise RuntimeError(f"An error occurred while generating embeddings: {e}")

        return embeddings, source_indices, split_info

    def validate_keys(self, keys_to_validate, keys_validation, keys_to_validate_name, keys_validation_name):
        for key in keys_to_validate:
            if key not in keys_validation:
                raise ValueError(f"Invalid key '{key}' in {keys_to_validate_name} not found in {keys_validation_name}.")
                    
    def collect_document_keys(self, documents):
        """
        Collect all keys from all documents, including nested ones, and return them as a list.
        """
        document_keys = set()  # Use a set to collect unique keys

        def collect_keys(d, key_prefix):
            if isinstance(d, dict):
                for key, value in d.items():
                    full_key = f"{key_prefix}.{key}" if key_prefix else key
                    document_keys.add(full_key)  # Move this line out of the else block
                    if isinstance(value, dict):
                        collect_keys(value, full_key)
                    elif isinstance(value, list):
                        collect_keys(value, full_key)
            elif isinstance(d, list):
                for i, item in enumerate(d):
                    index_key = f"{key_prefix}[{i}]"
                    document_keys.add(index_key)  # Move this line out of the else block
                    if isinstance(item, dict):
                        collect_keys(item, index_key)
                    elif isinstance(item, list):
                        collect_keys(item, index_key)

        for document in documents:
            collect_keys(document, "")

        return list(document_keys)  # Convert set to list before returning

    def compile_keys(self):
        self.compiled_keys = []
        if isinstance(self.select_keys, str):  # Handle single string case
            self.compiled_keys.append(self.select_keys)
        elif isinstance(self.select_keys, list):  # Handle list of strings case
            for k in self.select_keys:
                self.compiled_keys.append(k)

    def _store_metadata(self, document, unique_index):
        """Stores metadata for a given document."""
        if not isinstance(document, dict):
            return
        filtered_document = self.filter_document(document)
        metadata = {}
        for key in self.metadata_keys:
            if key == "timestamp":
                existing_timestamp = self._metadata_index.get(unique_index, {}).get("timestamp")
                if existing_timestamp is None and self.add_timestamp is True:
                    metadata[key] = float(datetime.datetime.now().timestamp())
                else:
                    metadata[key] = existing_timestamp
            else:
                nested_keys = key.split('.')
                value = self.get_nested_value(filtered_document, nested_keys)
                if value is not None:
                    metadata[key] = value
        if metadata:
            self._metadata_index[unique_index] = metadata

    def filter_document(self, document):
        """Filters a document based on keys."""
        if not self.compiled_keys or not isinstance(document, dict):
            return document
        
        filtered_doc = {}
        for full_key in self.compiled_keys:
            # Use regular expression to split the full_key while keeping square brackets intact
            nested_keys = re.split(r'\.|\[|\]', full_key)
            nested_keys = [key for key in nested_keys if key]  # Remove empty strings
            value = self.get_nested_value(document, nested_keys)
            if value is not None:
                filtered_doc[full_key] = value
                
        return filtered_doc if filtered_doc else document

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


    def size(self, with_chunks=False):
        """
        Returns the number of documents in the database.
        Args:
            with_chunks (bool): If True, include the chunks in the count. Default is False.
        """
        if with_chunks:
            return len(self.documents)
        else:
            # Count the occurrences of each index in source_indices to find unique documents
            return len(Counter(self.source_indices))

    def dict(self, vectors=False):
        """
        Returns the database in dictionary format.
        Args:
            vectors (bool): If True, include vectors in the output.
        """
        try:
            if not self.source_indices:
                print("Debug: source_indices is empty.")
                return []
            if not self.documents:
                print("Debug: documents is empty.")
                return []
            
            if len(self.source_indices) != len(self.documents):
                print(f"Debug: Inconsistency between length of source_indices {len(self.source_indices)} and documents {len(self.documents)}.")
                return []

            # Count the occurrences of each index in source_indices
            index_counts = Counter(self.source_indices)
            
            output = []
            i = 0  # Start index for self.documents
            unique_index = 0  # Counter for unique documents
            
            for source_index, count in sorted(index_counts.items()):
                if i >= len(self.documents):
                    print(f"Debug: Index i={i} is out of range for self.documents.")
                    break
                
                doc = self.documents[i]
                
                if vectors:
                    if i >= len(self.vectors):
                        print(f"Debug: Index i={i} is out of range for self.vectors.")
                        break
                    
                    vec = self.vectors[i].tolist()
                    output.append({"document": doc, "vector": vec, "index": unique_index})
                else:
                    output.append({"document": doc, "index": unique_index})
                    
                i += count  # Move the start index to the next unique document
                unique_index += 1  # Increment the unique document index

            return output

        except Exception as e:
            print(f"Error while generating dictionary: {e}")
            return []

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
            filtered_documents = [self.filter_document(doc) for doc in documents]
            self.add_documents(filtered_documents, vectors, add_timestamp)
        else:
            filtered_document = self.filter_document(documents)
            self.add_document(filtered_document, vectors, add_timestamp=add_timestamp)
            self.commit_pending()

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
        
        split_info = {}
        
        # Only add a timestamp if the document is a dictionary and add_timestamp is True
        if isinstance(document, dict) and add_timestamp:
            timestamp = datetime.datetime.now().timestamp()
            # Add timestamp to the document's metadata
            if 'metadata' not in document:
                document['metadata'] = {}
            document['metadata']['timestamp'] = float(timestamp)  # Store timestamp as float
        
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
        
        # Calculate the unique index for this document for metadata storage
        unique_index = len(self.documents) + len(self.pending_documents) - 1
        self._store_metadata(document, unique_index)  # Store metadata


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
            if vectors is None:
                vectors, source_indices, split_info = self.embedding_function(documents)
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
        
        # Reindex source_indices
        for i, idx in enumerate(self.source_indices):
            self.source_indices[i] -= len([d for d in indices if d < idx])

    def save(self, storage_file, format='pickle'):        
        # Check if there's nothing to save
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            print("Nothing to save. Exit.")
            return
        data = {
            "vectors": [vector.tolist() for vector in self.vectors],
            "documents": self.documents,
            "source_indices": self.source_indices,
            "split_info": self.split_info,
            "metadata_index": self._metadata_index
        }
        
        if format == 'pickle':
            self._save_pickle(storage_file, data)
        elif format == 'json':
            self._save_json(storage_file, data)
        elif format == 'sqlite':
            self._save_sqlite(storage_file, data)
        else:
            raise ValueError(f"Unsupported format '{format}'")
            
    def _save_pickle(self, storage_file, data):
        try:
            if storage_file.endswith(".gz"):
                with gzip.open(storage_file, "wb") as f:
                    pickle.dump(data, f)
            else:
                with open(storage_file, "wb") as f:
                    pickle.dump(data, f)
        except Exception as e:
            raise RuntimeError(f"An exception occurred during pickle save: {e}")
            
    def _save_json(self, storage_file, data):
        try:
            with open(storage_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            raise RuntimeError(f"An exception occurred during JSON save: {e}")
            
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
                
                # Insert data into documents and vectors
                for doc, vec in zip(data["documents"], data["vectors"]):
                    cursor.execute('INSERT INTO documents (data) VALUES (?)', (json.dumps(doc),))
                    doc_id = cursor.lastrowid
                    cursor.execute('INSERT INTO vectors (document_id, vector) VALUES (?, ?)', (doc_id, json.dumps(vec)))
                
                # Insert data into source_indices and split_info 
                for index in data["source_indices"]:
                    cursor.execute('INSERT INTO source_indices (value) VALUES (?)', (index,))
                    
                cursor.execute('INSERT INTO split_info (value) VALUES (?)', (json.dumps(data["split_info"]),))
                
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"SQLite error during save: {e}")

    def load(self, storage_file, format='pickle'):
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
            self._metadata_index = data.get("metadata_index", {})
            self.split_info = data.get("split_info", {})

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
        with closing(sqlite3.connect(storage_file)) as conn:  # Ensures the connection is closed
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

                return {
                    "vectors": vectors,
                    "documents": documents,
                    "source_indices": source_indices,
                    "split_info": split_info
                }
            
            except sqlite3.Error as e:
                raise RuntimeError(f"SQLite error during load: {e}")
        
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
            pattern = re.compile(r'[\[\].]')
            for key in keys:
                key_parts = [k for k in pattern.split(key) if k]
                for part in key_parts:
                    if value is None:
                        break
                    if part.isdigit():
                        index = int(part)
                        value = value[index] if index < len(value) and isinstance(value, list) else None
                    elif isinstance(value, dict):
                        value = value.get(part, None)
                    elif isinstance(value, list):
                        value = [sub_value.get(part, None) for sub_value in value if isinstance(sub_value, dict)]
                    else:
                        value = None
            return value
        except (KeyError, TypeError, AttributeError, IndexError):
            return None


    def filter_by_key(self, vectors, documents, keys):
        if not isinstance(keys, (list, tuple)):
            keys = [keys]

        # Validate the Keys using the generic validate method
        self.validate_keys(keys, self.document_keys, "query_keys", "document_keys")

        filtered_vectors_dict = {}
        filtered_documents_dict = {}

        # Pre-Process Keys
        processed_keys = [key.split('.') if '.' in key else [key] for key in keys]

        for vec, doc, doc_id in zip(vectors, documents, map(id, documents)):
            if not isinstance(doc, dict):
                continue  # Skip non-dict documents

            vectors_for_keys = []
            for key in processed_keys:
                sub_text = self.get_nested_value(doc, key)
                if sub_text is not None:
                    vector_for_key = self.embedding_function([str(sub_text)])[0].flatten()
                else:
                    vector_for_key = np.zeros(self.vectors.shape[1])  # Assuming vector dimensionality is self.vectors.shape[1]
                vectors_for_keys.append(vector_for_key)
            if not vectors_for_keys:
                continue

            # Average the vectors obtained for each key
            averaged_vector = sum(vectors_for_keys) / len(vectors_for_keys)
            # Optimize Vector Averaging (if needed)
            if doc_id not in filtered_vectors_dict:
                filtered_vectors_dict[doc_id] = (averaged_vector, 1)  # Store vector and count
                filtered_documents_dict[doc_id] = doc
            else:
                existing_vec, count = filtered_vectors_dict[doc_id]
                new_averaged_vec = (existing_vec * count + averaged_vector) / (count + 1)
                filtered_vectors_dict[doc_id] = (new_averaged_vec, count + 1)
                
        filtered_vectors = [vec for vec, _ in filtered_vectors_dict.values()]
        filtered_documents = list(filtered_documents_dict.values())

        return filtered_vectors, filtered_documents

    def generate_query_vector(self, query_text):
        query_vector = self.embedding_function([query_text])

        if not query_vector:
            raise ValueError("Failed to generate an embedding for the query text.")
        return query_vector[0]

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

    def recursive_sentence_filter(self, obj, sentence_filter):
        """
        Recursively traverse an object to find the sentence filter, considering whole words.
        """
        def tokenize(text):
            """
            Tokenizes a string into words, removing punctuation.
            """
            text = ''.join(c for c in text if c not in string.punctuation)
            return re.findall(r'\b\w+\b', text.lower())

        if isinstance(obj, dict):
            return any(self.recursive_sentence_filter(v, sentence_filter) for v in obj.values())
        elif isinstance(obj, list):
            return any(self.recursive_sentence_filter(v, sentence_filter) for v in obj)
        elif isinstance(obj, str):
            # Tokenize the object string and the sentence filter
            obj_tokens = set(tokenize(obj))
            sentence_filter_tokens = set(tokenize(sentence_filter))
            # Check if all tokens in the sentence filter are present in the object string
            return sentence_filter_tokens.issubset(obj_tokens)
        else:
            return False

    def filter_by_sentence(self, vectors, documents, sentence_filters):
        if not isinstance(sentence_filters, (list, tuple)):
            sentence_filters = [sentence_filters]  # Ensure sentence_filters is a list or tuple

        filtered_vectors = []
        filtered_documents = []

        for vec, doc in zip(vectors, documents):
            # Check if all sentence filters are found in the document
            if all(self.recursive_sentence_filter(doc, sentence_filter) for sentence_filter in sentence_filters):
                filtered_vectors.append(vec)
                filtered_documents.append(doc)

        return filtered_vectors, filtered_documents

    def _generate_and_validate_query_vector(self, query_input):
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

    def _filter_by_metadata(self, metadata_filter):
        """
        A new helper method to filter vectors and documents by metadata.
        """
        # Validate the metadata keys
        self.validate_keys(metadata_filter.keys(), self.metadata_keys, "metadata_filter", "metadata_keys")
        
        # Initialize an empty list to hold the indices of documents that match the metadata filter
        filtered_indices = []
        
        # Loop through each item in the metadata index to find matches
        for idx, metadata in self._metadata_index.items():
            # Assume the document matches until proven otherwise
            is_match = True
            
            # Loop through each key-value pair in the metadata filter to check for matches
            for key, value in metadata_filter.items():
                # Check if the key exists in the metadata and if its value matches the filter
                if metadata.get(key) != value:
                    is_match = False
                    break  # No need to check further for this document
                    
            # If the document matches all key-value pairs in the filter, add its index to the list
            if is_match:
                filtered_indices.append(idx)
        # Check for length mismatch between self.vectors and self.source_indices
        if len(self.vectors) != len(self.source_indices):
            raise Exception("Length mismatch between vectors and source_indices.")
            
        # Use the filtered indices to get the corresponding vectors and documents
        filtered_vectors = [vec for i, vec in enumerate(self.vectors) if self.source_indices[i] in filtered_indices]
        filtered_documents = [self.documents[i] for i in filtered_indices]

        return np.array(filtered_vectors, dtype=self.fp_precision), filtered_documents

    def _apply_filters(self, filters):
        filtered_vectors = self.vectors
        filtered_documents = self.documents
        filtered_doc_ids = set(id(doc) for doc in self.documents)
        skip_doc_value = None
        skip_doc_position = None

        for i, (filter_name, filter_params) in enumerate(filters):
            if filter_name not in ['key', 'metadata', 'sentence', 'skip_doc']:
                raise ValueError(f"Invalid filter name {filter_name}")

            # Filter by key
            if filter_name == 'key':
                filtered_vectors, filtered_documents_by_key = self.filter_by_key(filtered_vectors, filtered_documents, filter_params)

            # Filter by metadata
            elif filter_name == 'metadata':
                if not self.metadata_keys:
                    raise ValueError("The 'metadata_keys' parameter has not been set in HyperDB(). Cannot filter by metadata.")
                _, filtered_documents_by_metadata = self._filter_by_metadata(filter_params)

            # Filter by sentence
            elif filter_name == 'sentence':
                _, filtered_documents_by_sentence = self.filter_by_sentence(self.vectors, self.documents, filter_params)

            # Filter by skip_doc
            elif filter_name == 'skip_doc':
                skip_doc_value = filter_params
                skip_doc_position = i
                continue

            # Update the set of filtered document IDs based on this filter
            current_filtered_ids = set(id(doc) for doc in locals().get(f"filtered_documents_by_{filter_name}"))
            filtered_doc_ids &= current_filtered_ids  # Take the intersection with the existing set
            
        # Apply skip_doc filter if it was specified at the beginning
        if skip_doc_position == 0:
            filtered_vectors, filtered_documents = self.apply_skip_doc(self.vectors, self.documents, skip_doc_value)

        # Filter the vectors and documents based on the intersection of all filters
        filtered_vectors = [vec for vec, doc in zip(filtered_vectors, filtered_documents) if id(doc) in filtered_doc_ids]
        filtered_documents = [doc for doc in self.documents if id(doc) in filtered_doc_ids]

        # Apply skip_doc filter if it was specified at the end
        if skip_doc_position == len(filters) - 1:
            filtered_vectors, filtered_documents = self.apply_skip_doc(filtered_vectors, filtered_documents, skip_doc_value)

        return filtered_vectors, filtered_documents


    def query(self, query_input, top_k=5, return_similarities=True, filters=None, recency_bias=0, timestamp_key=None, metric='cosine_similarity'):  
        """
        Query the document store to retrieve relevant documents based on a variety of optional parameters.
        
        Parameters:
        - query_input (str or array-like): The query as a string or as a vector.
        - top_k (int): The number of top matches to return.
        - return_similarities (bool): Whether to return similarity scores along with documents.
        - filters (list): A list of tuples, where each tuple contains a filter name and its parameters. The order can be chosen by the user.
        - recency_bias (float): A factor to bias toward more recent documents. Needs a timestamp key to be present in 'metadata_keys'.
        - timestamp_key (str): A specific key to use for the recency_bias factor in the similarity search.
        - metric (str): The metric used to compute similarity. (default = 'cosine_similarity')
            Supported values: 'dot_product', 'cosine_similarity', 'euclidean_metric', 
            'manhattan_distance', 'jaccard_similarity', 'pearson_correlation', 
            'mahalanobis_distance', 'hamming_distance'.
        """    
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            raise Exception("The database is empty. Cannot proceed with the query.")

        # Validate the metric
        valid_metrics = [
            'dot_product', 'cosine_similarity', 'euclidean_metric', 'manhattan_distance',
            'jaccard_similarity', 'pearson_correlation', 'mahalanobis_distance', 'hamming_distance'
        ]
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric {metric}")
        try:
            query_vector = self._generate_and_validate_query_vector(query_input)    
            
            filtered_vectors = self.vectors
            filtered_documents = self.documents
            if len(filtered_vectors) != len(filtered_documents):
                print(f"Inconsistency detected between filtered vectors {len(filtered_vectors)} and filtered documents {len(filtered_documents)}.")
                return []
            
            # Apply filters based on user input
            # Initialize a set to hold the identifiers of the filtered documents
            filtered_doc_ids = set(id(doc) for doc in self.documents)

            # Initialize skip_doc_value and skip_doc_position
            skip_doc_value = None
            skip_doc_position = None

            if filters:
               filtered_vectors, filtered_documents = self._apply_filters(filters)

            if len(filtered_vectors) == 0:
                print("No document matches your query.")
                return []

            # Handle timestamp retrieval based on recency_bias and timestamp_key
            if recency_bias != 0 and timestamp_key is None:
                if "timestamp" not in self.metadata_keys:
                    raise ValueError("When recency_bias is not 0, the 'timestamp' key must be present in metadata_keys if timestamp_key is not provided.")
                timestamp_key = "timestamp"  # Use 'timestamp' as the default key if recency_bias is not 0

            if timestamp_key:
                if timestamp_key not in self.metadata_keys:
                    raise ValueError(f"The timestamp_key '{timestamp_key}' is not in metadata_keys.")

                # Check if the timestamp_key is nested or at the root level
                if '.' in timestamp_key:
                    nested_keys = timestamp_key.split('.')
                else:
                    nested_keys = [timestamp_key]

                timestamps = [self.get_nested_value(document, nested_keys) for document in filtered_documents]

                if None in timestamps:
                    raise ValueError("All timestamps must be populated when recency_bias is not 0 or timestamp_key is provided.")
                    
                timestamps = np.array(timestamps, dtype=float)
            else:
                timestamps = None

            if top_k > len(filtered_documents):
                print(f"Warning: top_k ({top_k}) is greater than the number of filtered documents ({len(filtered_documents)}). Setting top_k to {len(filtered_documents)}.")
                top_k = len(filtered_documents)

            ranked_results, scores = ranking.hyperDB_ranking_algorithm_sort(
                filtered_vectors, query_vector, top_k=top_k, metric=metric, timestamps=timestamps, recency_bias=recency_bias
            )

            if max(ranked_results) >= len(filtered_documents):
                raise IndexError(f"Invalid index in ranked_results. Max index: {max(ranked_results)}, Length of filtered_documents: {len(filtered_documents)}")

            if return_similarities:
                return list(zip([filtered_documents[index] for index in ranked_results], scores))
            
            return [filtered_documents[index] for index in ranked_results]

        except (ValueError, TypeError) as e:
            print(f"An exception occurred due to invalid input: {e}")
            raise e
        except Exception as e:
            print(f"An unknown exception occurred: {e}")
            raise
