import gzip
import pickle
import json
import sqlite3
import datetime
import numpy as np
import collections
import string
import torch
from contextlib import closing
from transformers import BertTokenizer
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

from hyperdb.ranking_algorithm import (
    get_norm_vector,
    dot_product,
    cosine_similarity,
    euclidean_metric,
    hyper_SVM_ranking_algorithm_sort,
    custom_ranking_algorithm_sort
)

EMBEDDING_MODEL = None
tokenizer = None
MAX_LENGTH = 256

#Maximum max_length=256 for all-MiniLM-L6-v2
def text_to_chunks(text, tokenizer, max_length=MAX_LENGTH):
    tokens = tokenizer.encode(text, truncation=False)
    chunks = []

    for i in range(0, len(tokens), max_length):
        chunk_tokens = tokens[i:i+max_length]
        chunk = tokenizer.decode(chunk_tokens, clean_up_tokenization_spaces=True)
        chunks.append(chunk)

    return chunks

def get_embedding(documents, fp_precision=np.float32):
    """Embedding function that uses Sentence Transformers."""
    global EMBEDDING_MODEL, tokenizer
    try:
        if EMBEDDING_MODEL is None or tokenizer is None:
            # Automatically select the GPU if available, otherwise use CPU
            device = 'cuda' if torch.cuda.is_available() else 'cpu'
            EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device=device)
            tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    except Exception as e:
        raise RuntimeError(f"Failed to initialize the Sentence Transformer model: {e}")

    if documents is None:
        raise ValueError("Documents cannot be None.")
        
    texts = []
    source_indices = []
    split_info = {}

    try:
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
            if isinstance(documents[0], dict):
                for i, doc in enumerate(documents):
                    for k, value in doc.items():
                        if not isinstance(value, str):
                            try:
                                value = str(value)
                            except Exception as e:
                                raise ValueError(f"Failed to convert key '{k}' to string: {e}")
                        chunks = text_to_chunks(value, tokenizer)
                        if len(chunks) > 1:
                            split_info[i] = True
                        texts.extend(chunks)
                        source_indices.extend([i]*len(chunks))
            elif isinstance(documents[0], str):
                texts = documents
                source_indices = list(range(len(documents)))
            else:
                raise ValueError("Unsupported document type.")
        else:
            raise ValueError("Documents should either be a string or a list.")
        embeddings = EMBEDDING_MODEL.encode(texts).astype(fp_precision)  
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
        similarity_metric="cosine",
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
        self.key = key
        self.compiled_keys = None
        if self.key:
            self.compile_keys()
        self.fp_precision = getattr(np, fp_precision)   # Convert the string to a NumPy dtype
        
        self.embedding_function = embedding_function or (
            lambda docs: get_embedding(docs, fp_precision=fp_precision)
        )
       
        self.pending_vectors = []  # Temporary storage for new vectors
        self.pending_documents = []  # Temporary storage for new documents
        self.pending_source_indices = []  # Temporary storage for new source indices
       
            
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
        else:
            raise Exception("Similarity metric not supported. Please use either 'dot', 'cosine', 'euclidean'.")

    def commit_pending(self):
        """Commit the pending vectors and documents to the main storage."""
        if self.pending_vectors:
            new_vectors = np.vstack(self.pending_vectors)
            if self.vectors is None:
                self.vectors = new_vectors
            else:
                self.vectors = np.vstack([self.vectors, new_vectors])

            self.documents.extend(self.pending_documents)
            self.source_indices.extend(self.pending_source_indices)
            
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
        document = self.filter_document(document)
        if vectors is None:
            vectors, _, _ = self.embedding_function([document])

        if vectors.size == 0:  # Check if vectors is empty
            raise ValueError("No vectors returned by the embedding_function.")

        if len(vectors.shape) == 1:
            vectors = np.expand_dims(vectors, axis=0)

        if len(vectors.shape) != 2:
            raise ValueError("Vectors does not have the expected structure.")

        # Append to pending vectors and documents
        for _ in range(count):
            self.pending_vectors.append(vectors)
            self.pending_documents.append(document)
            self.pending_source_indices.append(len(self.documents) + len(self.pending_documents) - 1)


        # Only add a timestamp if the document is a dictionary and add_timestamp is True
        if isinstance(document, dict) and add_timestamp:
            timestamp = datetime.datetime.now().timestamp()
            document['timestamp'] = str(timestamp)

        self.documents.extend([document]*count)  # Extend the document list with the same document for all chunks
        self.source_indices.extend([len(self.documents) - 1]*count)  # Extend the source_indices list with the same index for all chunks


    def add_documents(self, documents, vectors=None, add_timestamp=False):
        """
        Add multiple documents to the database.
        Args:
            documents (list): A list of documents to add.
            vectors (list): Pre-computed vectors for the documents. If provided, should match the length and order of documents.
        """
        if not documents:
            return

        filtered_documents = [self.filter_document(doc) for doc in documents]
        if vectors is None:
            vectors, source_indices, split_info = self.embedding_function(filtered_documents)
        else:
            source_indices = list(range(len(documents)))

        self.pending_source_indices.extend(source_indices)

        for i, document in enumerate(documents):
            count = split_info.get(i, 1)
            self.add_document(document, vectors[i], count, add_timestamp=False)

        # Commit the pending vectors and documents to the main storage
        self.commit_pending()
  
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
                return

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
        
        for row in cursor.execute('SELECT data FROM documents'):
            documents.append(json.loads(row[0]))

        for row in cursor.execute('SELECT vector FROM vectors ORDER BY document_id'):
            vectors.append(json.loads(row[0]))
 
        conn.close()
        return {"vectors": vectors, "documents": documents}
        
    def compute_and_save_word_frequencies(self, output_file_path):
        """
        Compute word frequencies from the documents in the database and save them to a text file.
        
        Args:
            output_file_path (str): The path to the output text file.
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
                doc_id = id(doc)  # Using the object id as a unique identifier
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
                    if doc_id in filtered_vectors_dict:
                        filtered_vectors_dict[doc_id].append(new_vec)
                    else:
                        filtered_vectors_dict[doc_id] = [new_vec]
                        filtered_documents_dict[doc_id] = doc

        # Average the vectors for each document
        for doc_id, vec_list in filtered_vectors_dict.items():
            filtered_vectors_dict[doc_id] = np.mean(np.array(vec_list), axis=0)
        
        filtered_vectors = np.array(list(filtered_vectors_dict.values()))
        filtered_documents = list(filtered_documents_dict.values())
        
        return filtered_vectors, filtered_documents


    def query(self, query_text, top_k=5, return_similarities=True, key=None, recency_bias=0, use_timestamp=False, timestamp_key='timestamp', skip_doc=0):
        # Check if there's nothing to query
        if self.vectors is None or self.vectors.size == 0 or not self.documents:
            return []
        
        try:
            query_vector = self.embedding_function([query_text])
            if not query_vector:
                raise ValueError("Failed to generate an embedding for the query text.")
            query_vector = query_vector[0]
            filtered_vectors = []
            filtered_documents = []

            if key and all(isinstance(doc, dict) for doc in self.documents):
                filtered_vectors, filtered_documents = self.filter_by_key(self.vectors, self.documents, key)
                if len(filtered_vectors) == 0:
                    print(f"Warning: No documents were filtered using the key '{key}'. It might be non-existent or have null values.")
                
            if len(filtered_vectors) == 0:
                filtered_vectors = self.vectors
                filtered_documents = self.documents

            # Before running the ranking logic, apply skip_doc
            if abs(skip_doc) > len(filtered_documents):
                print(f"Warning: The absolute value of skip_doc ({abs(skip_doc)}) is greater than the total number of documents ({len(filtered_documents)}).")
            if skip_doc > 0:
                filtered_vectors = filtered_vectors[skip_doc:]
                filtered_documents = filtered_documents[skip_doc:]
            elif skip_doc < 0:
                filtered_vectors = filtered_vectors[:skip_doc]
                filtered_documents = filtered_documents[:skip_doc]

            # Convert to NumPy array for computation
            filtered_vectors = np.array(filtered_vectors, dtype=self.fp_precision)

            # Decide which ranking algorithm to use based on the use_timestamp flag
            if use_timestamp:
                timestamps = []
                for document in filtered_documents:
                    if timestamp_key in document:
                        timestamps.append(document.get(timestamp_key, 0))
                    else:
                        print(f"Warning: Missing timestamp_key '{timestamp_key}' in one of the documents. Using a default value of 0.")
                        timestamps.append(0)

                ranked_results, combined_scores, original_similarities = custom_ranking_algorithm_sort(
                    filtered_vectors, query_vector, timestamps, top_k=top_k, metric=self.similarity_metric, recency_bias=recency_bias
                )
            else:
                ranked_results, original_similarities = hyper_SVM_ranking_algorithm_sort(
                    filtered_vectors, query_vector, top_k=top_k, metric=self.similarity_metric
                )
                combined_scores = original_similarities
            
            original_similarities = get_norm_vector(original_similarities)
            combined_scores = get_norm_vector(combined_scores)
            if return_similarities:
                return list(
                    zip([filtered_documents[index] for index in ranked_results], combined_scores, original_similarities)
                )
            
            return [filtered_documents[index] for index in ranked_results]

        except (ValueError, TypeError) as e:
            print(f"An exception occurred due to invalid input: {e}")
            return []
        except Exception as e:
            print(f"An unknown exception occurred: {e}")
            return []
