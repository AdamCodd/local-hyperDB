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
import os
import onnxruntime as ort
import hyperdb.ranking_algorithm as ranking
import cachetools
from pympler import asizeof
from itertools import repeat
from contextlib import closing
from transformers import BertTokenizerFast
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer
from annoy import AnnoyIndex
from collections.abc import Iterable

ort.set_default_logger_severity(3) # Disable onnxruntime useless warnings when switching to GPU
EMBEDDING_MODEL = None
tokenizer = None
MAX_LENGTH = 510 # 512 - 2 to account for special tokens added by the BERT tokenizer
NESTED_PATTERN = re.compile(r'[\[\].]')

class HyperDB:
    """
    HyperDB is a class for efficient document storage and similarity search.
    
    Args:
      - documents (list): List of documents to be added to the database.
      - vectors (array-like): Precomputed vectors for the documents.
      - select_keys (list): Keys to be selected from the documents to embed only a part of it.
      - embedding_function (callable): Function to generate embeddings for documents.
      - fp_precision (str): Floating-point precision for numerical operations. Can be "float16", "float32", or "float64".
      - add_timestamp (bool): Whether to add a timestamp to each document.
      - metadata_keys (list): List of keys for metadata used later for filtering.
      - ann_metric (str): The metric used for Approximate Nearest Neighbor (ANN) search. 
                             Accepted values are "angular", "euclidean", "manhattan", "hamming", and "dot".
      - n_trees (int): The number of trees to be used in the ANN index for fast querying. Default is 10.
    """
    def __init__(
        self,
        documents=None,
        vectors=None,
        select_keys=None,
        embedding_function=None,
        fp_precision="float32",
        add_timestamp=False,
        metadata_keys=None,
        ann_metric="cosine",
        n_trees=10,
        cache_size=256
    ):
    
        #LRU Cache
        self.lru_cache = cachetools.LRUCache(maxsize=cache_size)
        self.cache_hits = 0
        self.cache_misses = 0
    
        # Validate floating-point precision
        if fp_precision not in ["float16", "float32", "float64"]:
            raise ValueError("Unsupported floating-point precision.")

        # Validate ann_metric
        accepted_metrics = ["angular", "euclidean", "manhattan", "hamming", "dot", "cosine"]
        if ann_metric not in accepted_metrics:
            raise ValueError(f"Unsupported ANN metric. Accepted values are: {', '.join(accepted_metrics)}")
            
        self.source_indices = []
        self.split_info = {}
        self.documents = []
        self.vectors = None
        self.select_keys = select_keys
        self.add_timestamp = add_timestamp
        
        self.fp_precision = getattr(np, fp_precision)     
        self.initialize_model()
        self.embedding_function = embedding_function or self.get_embedding
        self.n_trees = n_trees  # Store the n_trees value for later use
        
        if isinstance(self.select_keys, str):
            self.select_keys = [self.select_keys]

        # Normalized vectors for ANN using cosine metric
        self.vectors_normalized = False
        
        # Initialize temporary storage for new vectors, documents, and source indices
        self.pending_vectors = []
        self.pending_documents = []
        self.pending_source_indices = []

        self._metadata_index = {}  # Key will be the unique doc index, value will be the metadata dictionary
        self.metadata_keys = metadata_keys or [] # Store the metadata keys  
  
        if isinstance(metadata_keys, str):
            self.metadata_keys = [metadata_keys]
  
        self.document_keys = []
  
        # Add 'timestamp' to metadata_keys if add_timestamp is True
        if self.add_timestamp and "timestamp" not in self.metadata_keys:
            self.metadata_keys.append("timestamp")
            self.document_keys.append("timestamp")

        # Validate and convert documents
        if documents:
            documents = self.validate_and_convert_documents(documents)
    
        # Assuming all documents have the same structure, we collect all unique keys
        if documents and isinstance(documents[0], dict):
           self.document_keys = self.collect_document_keys(documents)
           if self.metadata_keys:
              if self.select_keys:
                self.validate_keys(self.metadata_keys, self.select_keys, "metadata_keys", "select_keys")
              self.validate_keys(self.metadata_keys, self.document_keys, "metadata_keys", "document_keys")

        # Initialize ANN index
        self.ann_metric = ann_metric
        self.ann_index = None
        self.ann_dim = None  # Dimension of the vectors, will be set when adding first document

        # If vectors are provided, use them; otherwise, add documents using `add` method
        if vectors is not None:
            self.validate_vector_uniformity(vectors)
            self.vectors = vectors
            self.documents = documents
            if self.select_keys:
                self.documents = [self.filter_document(doc) for doc in self.documents]
            self.source_indices = list(range(len(documents)))
            self._build_ann_index()
        elif documents:
            self.add(documents, vectors=None, add_timestamp=self.add_timestamp)

    def validate_vector_uniformity(self, vectors):
        """
        Validates that all vectors have the same dimension.
        Args:
            vectors (array-like): Array of vectors to validate.
        """
        if len(vectors) == 0:
            raise ValueError("The vector array is empty.")

        first_vector_length = len(vectors[0])
        if not all(len(vec) == first_vector_length for vec in vectors):
            raise ValueError("All vectors must have the same dimension.")

        self.ann_dim = first_vector_length  # Set the dimensionality of vectors for the ANN index

    def validate_and_convert_documents(self, documents):
            """
            Validates and converts documents to dictionary format if they are not already.
            
            Args:
              - documents (list or str or dict): Document(s) to be validated and converted.
            """
            # Check for basic supported types
            if isinstance(documents, (list, tuple, str, dict)):
                # Process based on type
                if isinstance(documents, (list, tuple)):
                    # Process multiple documents
                    validated_documents = [
                        {'document': doc} if not isinstance(doc, dict) else doc
                        for doc in documents
                    ]
                else:
                    # Process a single document
                    validated_documents = [{'document': documents}] if not isinstance(documents, dict) else [documents]
            # Check for other iterable types and handle them
            elif isinstance(documents, Iterable) and not isinstance(documents, (str, bytes)):
                # Process iterable documents (similar to list/tuple)
                validated_documents = [
                    {'document': doc} if not isinstance(doc, dict) else doc
                    for doc in documents
                ]
            else:
                # Unsupported type
                raise ValueError(f"Unsupported document type: {type(documents)}. Expected list, tuple, or dict.")

            return validated_documents

    def _build_ann_index(self):
        if self.vectors is None or self.vectors.shape[0] == 0:
            return
        
        # Use the stored n_trees value instead of calculating it
        n_trees = self.n_trees
        
        if self.ann_metric == 'cosine':
            # Normalize vectors for cosine
            vectors_to_add = ranking.get_norm_vector(self.vectors)
            self.vectors_normalized = True
            annoy_metric = 'euclidean'
        else:
            vectors_to_add = self.vectors
            self.vectors_normalized = False
            annoy_metric = self.ann_metric
        
        # Use the stored ann_metric
        self.ann_index = AnnoyIndex(self.ann_dim, annoy_metric)
        
        for i, vec in enumerate(vectors_to_add):
            self.ann_index.add_item(i, vec)
        self.ann_index.build(n_trees=n_trees)

    def _update_ann_index(self):
        self._build_ann_index()

    def set_ann_metric(self, new_metric):
        """
        Set a new Annoy metric and rebuild the index.
        
        Args:
            new_metric (str): The new metric for Annoy ("angular", "euclidean", "manhattan", "hamming", "dot", "cosine")
        """
        if self.ann_metric != new_metric:
            self.ann_metric = new_metric
            self.vectors_normalized = False
        self._update_ann_index()

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
        if not self.select_keys or not isinstance(document, dict):
            return document
        
        filtered_doc = {}
        for full_key in self.select_keys:
            # Use regular expression to split the full_key while keeping square brackets intact
            nested_keys = re.split(r'\.|\[|\]', full_key)
            nested_keys = [key for key in nested_keys if key]  # Remove empty strings
            value = self.get_nested_value(document, nested_keys)
            if value is not None:
                filtered_doc[full_key] = value
                
        return filtered_doc if filtered_doc else document

    def size(self, with_chunks=False, metadata=None):
        """
        Returns the number of documents in the database, optionally filtering by metadata.
        
        Args:
            with_chunks (bool): If True, include the chunks in the count. Default is False.
            metadata (dict): A dictionary of {key: value} pairs to filter documents by metadata. Default is None.
        """
        if metadata:
            # Ensure that the metadata is a dictionary
            if not isinstance(metadata, dict):
                raise ValueError("metadata must be a dictionary of {key: value} pairs.")
            
            # Validate the metadata keys
            self.validate_keys(metadata.keys(), self.metadata_keys, "metadata", "metadata_keys")
            
            # Filter the documents based on the metadata
            _, filtered_documents = self._filter_by_metadata(metadata, self.vectors, self.documents)
            
            # Count the filtered documents
            if with_chunks:
                return len(filtered_documents)
            else:
                # When not considering chunks, count unique source indices of matching documents
                unique_indices = set(self.source_indices[self.documents.index(doc)] for doc in filtered_documents)
                return len(unique_indices)
        
        # The original functionality is preserved here if no metadata is provided.
        if with_chunks:
            return len(self.documents)
        else:
            # Count the occurrences of each index in source_indices to find unique documents
            return len(set(self.source_indices))

    def dict(self, vectors=False, metadata=None):
        """
        Returns the database in dictionary format, optionally filtering by metadata.
        
        Args:
            vectors (bool): If True, include vectors in the output.
            metadata (dict or tuple): Metadata to filter documents by. Can be a dictionary of {key: value} pairs or a tuple of (key, value).
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

            metadata_filter = {}
            # Apply metadata filter if provided
            if metadata:
                if isinstance(metadata, dict):
                    metadata_filter = metadata
                elif isinstance(metadata, tuple) and len(metadata) == 2:
                    metadata_filter = {metadata[0]: metadata[1]}
                else:
                    raise ValueError("metadata must be a dictionary of {key: value} pairs or a tuple of (key, value).")

                # Validate the metadata keys
                self.validate_keys(metadata_filter.keys(), self.metadata_keys, "metadata", "metadata_keys")
                
                # Filter the documents based on the metadata
                filtered_vectors, filtered_documents = self._filter_by_metadata(metadata_filter, self.vectors, self.documents)
            else:
                filtered_documents = self.documents
                filtered_vectors = self.vectors

            output = []
            for i, doc in enumerate(filtered_documents):
                doc_output = doc
                if vectors and filtered_vectors is not None:
                    doc_output["vector"] = filtered_vectors[i].tolist()
                output.append(doc_output)

            return output

        except Exception as e:
            print(f"Error while generating dictionary: {e}")
            return []

    def commit_pending(self):
        """Commit the pending vectors and documents to the main storage."""
        if not self.pending_vectors:
            return

        try:
            # Concatenate all pending vectors
            concatenated_pending_vectors = np.concatenate(self.pending_vectors, axis=0)
            if self.vectors is None:
                # If self.vectors is None, simply assign concatenated_pending_vectors to it
                self.vectors = concatenated_pending_vectors
            else:
                # Otherwise, concatenate the existing vectors with the new ones
                self.vectors = np.concatenate([self.vectors, concatenated_pending_vectors], axis=0)

            # Calculate total number of pending vectors (chunks)
            total_pending_vectors = sum(len(v) for v in self.pending_vectors)

            # Generate new source indices based on the chunks in self.split_info
            new_source_indices = []
            start_index = len(set(self.source_indices))
            
            chunk_sum = 0
            for j, (i, chunk_count) in enumerate(self.split_info.items(), start=start_index):
                if i >= start_index - 1:
                    new_source_indices.extend(repeat(j, chunk_count))  # More memory-efficient
                    chunk_sum += chunk_count
                    if chunk_sum == total_pending_vectors:
                        break

            # Consistency check
            if len(new_source_indices) != concatenated_pending_vectors.shape[0]:
                raise ValueError("Inconsistency detected in new source indices.")

            # Extend source indices and documents
            self.source_indices.extend(new_source_indices)
            self.documents.extend(self.pending_documents)

        except Exception as e:
            # Rollback transaction
            print(f"Error occurred during commit: {e}. Rolling back transaction.")
            # Restore the original self.vectors (before concatenation)
            if self.vectors is not None:
                self.vectors = self.vectors[:-len(concatenated_pending_vectors)]
            return

        # Cleanup
        self.pending_vectors.clear()
        self.pending_documents.clear()
        self.pending_source_indices.clear()


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
            self._update_ann_index()  # Update ANN index after committing
        self.clear_cache()

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
            for vector in vectors:  # Append each vector individually (for chunked documents)
                if len(vector.shape) == 1:
                    vector = np.expand_dims(vector, axis=0)
                temp_pending_vectors.append(vector)
            # Handle split_info
            for i, chunk_count in split_info.items():
                index = len(set(self.source_indices)) + len(temp_pending_documents) + i
                self.split_info[index] = chunk_count
            
        if vectors.size == 0:  # Check if vectors is empty
            raise ValueError("No vectors returned by the embedding_function.")
            
        if len(vectors.shape) == 1:
            vectors = np.expand_dims(vectors, axis=0)
            
        if len(vectors.shape) != 2:
            raise ValueError("Vectors does not have the expected structure.")

        # Set ann_dim by looking at the shape of the first vector only
        if self.ann_dim is None and vectors is not None and len(vectors.shape) == 2:
            self.ann_dim = vectors[0].shape[0]
        
        # Append to pending documents and source indices
        last_index = len(self.documents) + len(temp_pending_documents)
        for _ in range(count):
            temp_pending_documents.append(document)
            
            if split_info:
                for i, chunk_count in split_info.items():
                    temp_pending_source_indices.extend([last_index] * chunk_count)
            
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

            if len(vectors.shape) != 2:
                raise ValueError("Vectors does not have the expected structure.")

            # Set ann_dim by looking at the shape of the first vector only
            if self.ann_dim is None and vectors is not None and len(vectors.shape) == 2:
                self.ann_dim = vectors[0].shape[0]
            
            # Add to Pending Lists
            temp_pending_vectors = self.pending_vectors.copy()
            temp_pending_source_indices = self.pending_source_indices.copy()
            temp_pending_documents = self.pending_documents.copy()

            temp_pending_vectors.append(vectors)
            temp_pending_source_indices.extend(source_indices)

            vector_index = 0  # Track the current index in the vectors list
            for i, document in enumerate(documents):
                chunk_count = split_info.get(i, 1)  # Get chunk count for this document
                document_vectors = vectors[vector_index:vector_index + chunk_count]  # Get vectors for all chunks of this document
                self.add_document(document, document_vectors, chunk_count, add_timestamp)
                vector_index += chunk_count  # Move to the next set of vectors for the next document

            # Calculate total number of vectors in temp_pending_vectors
            total_vectors = sum(vec.shape[0] for vec in temp_pending_vectors)

            # Update self.split_info
            for i, chunk_count in split_info.items():
                index = len(set(self.source_indices)) + len(temp_pending_documents) + i
                self.split_info[index] = chunk_count

            # Commit to Main Storage
            if total_vectors == len(self.pending_documents):
                self.pending_vectors = temp_pending_vectors
                self.pending_source_indices = temp_pending_source_indices
                self.commit_pending()
                self._update_ann_index()  # Update ANN index after committing
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
        self._update_ann_index()  # Update ANN index after removal
        self.clear_cache()

    def save(self, storage_file, format='pickle', save_ann_index=True):        
        # Check if there's nothing to save
        if self.vectors is None or len(self.vectors) == 0 or not self.documents:
            print("Nothing to save. Exit.")
            return
        data = {
            "vectors": [vector.tolist() for vector in self.vectors],
            "documents": self.documents,
            "source_indices": self.source_indices,
            "split_info": self.split_info,
            "metadata_index": self._metadata_index,
            "vectors_normalized": self.vectors_normalized,
        }
        
        if format == 'pickle':
            self._save_pickle(storage_file, data)
        elif format == 'json':
            self._save_json(storage_file, data)
        elif format == 'sqlite':
            self._save_sqlite(storage_file, data)
        else:
            raise ValueError(f"Unsupported format '{format}'")
   
        # Save the ANN index if requested
        if save_ann_index and self.ann_index is not None:
            self._save_ann_index(storage_file)

    def _save_ann_index(self, storage_file):
        ann_index_file = str(storage_file) + '.ann'
        try:
            self.ann_index.save(ann_index_file)
        except Exception as e:
            raise RuntimeError(f"An exception occurred during ANN index save: {e}")
   
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
                # Create tables if they don't exist
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
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS metadata_index (
                    key TEXT PRIMARY KEY,
                    value TEXT
                )
                ''')
                
                cursor.execute('''
                CREATE TABLE IF NOT EXISTS settings (
                    name TEXT PRIMARY KEY,
                    value TEXT
                )
                ''')

                
                # Batch insert for documents
                document_data = [(json.dumps(doc),) for doc in data["documents"]]
                cursor.executemany('INSERT INTO documents (data) VALUES (?)', document_data)
                
                # Get last row id after batch insertion
                last_row_id = cursor.lastrowid
                
                # Batch insert for vectors with correct document_id
                vector_data = [(last_row_id - len(document_data) + i + 1, json.dumps(vec)) for i, vec in enumerate(data["vectors"])]
                cursor.executemany('INSERT INTO vectors (document_id, vector) VALUES (?, ?)', vector_data)

                # Batch insert for source_indices
                source_indices_data = [(index,) for index in data["source_indices"]]
                cursor.executemany('INSERT INTO source_indices (value) VALUES (?)', source_indices_data)
                
                # Insert for split_info (assuming only one row will be inserted)
                cursor.execute('INSERT INTO split_info (value) VALUES (?)', (json.dumps(data["split_info"]),))

                # Insert for metadata_index
                metadata_index_data = [(key, json.dumps(value)) for key, value in data["metadata_index"].items()]
                cursor.executemany('INSERT INTO metadata_index (key, value) VALUES (?, ?)', metadata_index_data)

                # Insert vectors_normalized
                cursor.execute('INSERT OR REPLACE INTO settings (name, value) VALUES (?, ?)', ('vectors_normalized', json.dumps(self.vectors_normalized)))
                
                conn.commit()
            except sqlite3.Error as e:
                conn.rollback()
                raise RuntimeError(f"SQLite error during save: {e}")


    def load(self, storage_file, format='pickle', load_ann_index=True, preload_ann_into_memory=False):
            if format == 'pickle':
                data = self._load_pickle(storage_file)
            elif format == 'json':
                data = self._load_json(storage_file)
            elif format == 'sqlite':
                data = self._load_sqlite(storage_file)
            else:
                raise ValueError(f"Unsupported format '{format}'")

            self.vectors = np.array(data["vectors"], dtype=self.fp_precision)
            
            # Set ann_dim based on the shape of the loaded vectors
            if len(self.vectors) > 0:
                self.ann_dim = self.vectors[0].shape[0]
            
            self.documents = data["documents"]
            self.source_indices = data.get("source_indices", [])
            self._metadata_index = data.get("metadata_index", {})
            self.split_info = data.get("split_info", {})
            self.vectors_normalized = data.get("vectors_normalized", False)
            
            # Load the ANN index if requested
            if load_ann_index and self.ann_dim is not None:
                self._load_ann_index(storage_file, preload_ann_into_memory)
        
    def _load_ann_index(self, storage_file, preload_ann_into_memory=True):
        ann_index_file = str(storage_file) + '.ann'
        annoy_metric = 'euclidean' if self.vectors_normalized else self.ann_metric
        try:
            # Check the size of the ANN file in bytes
            ann_file_size = os.path.getsize(ann_index_file)
            
            # Convert size to GB
            ann_file_size_gb = ann_file_size / (1024 ** 3)
            
            # Issue a warning if the ANN file size is greater than 2GB
            if ann_file_size_gb > 2 and preload_ann_into_memory:
                print("Warning: The ANN index file is {ann_file_size_gb:.2f}GB and may consume a lot of memory. Make sure your machine has enough available memory or set preload_ann_into_memory to False.")
            
            self.ann_index = AnnoyIndex(self.ann_dim, metric=annoy_metric)
            self.ann_index.load(ann_index_file, prefault=preload_ann_into_memory)
        except Exception as e:
            raise RuntimeError(f"An exception occurred during ANN index load: {e}")

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
            metadata_index = {}
            vectors_normalized = False

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

                # New data load for metadata_index
                for row in cursor.execute('SELECT key, value FROM metadata_index'):
                    metadata_index[row[0]] = json.loads(row[1])

                # New data load for vectors_normalized
                for row in cursor.execute('SELECT value FROM settings WHERE name = ?', ('vectors_normalized',)):
                    vectors_normalized = json.loads(row[0])


                return {
                    "vectors": vectors,
                    "documents": documents,
                    "source_indices": source_indices,
                    "split_info": split_info,
                    "metadata_index": metadata_index,
                    "vectors_normalized": vectors_normalized
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
        global NESTED_PATTERN
        try:
            value = dictionary
            for key in keys:
                key_parts = [k for k in NESTED_PATTERN.split(key) if k]
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
        if self.select_keys:
            self.validate_keys(keys, self.select_keys, "query_keys", "select_keys")

        filtered_vectors_dict = {}
        filtered_documents_dict = {}

        # Pre-Process Keys
        processed_keys = [key.split('.') if '.' in key else [key] for key in keys]

        for vec, doc in zip(vectors, documents):
            if not isinstance(doc, dict):
                continue  # Skip non-dict documents

            # Calculate the document ID only when it's needed
            doc_id = id(doc)

            vectors_for_keys = []
            for key in processed_keys:
                sub_text = self.get_nested_value(doc, key)
                if sub_text is not None:
                    vector_for_key = self.embedding_function([str(sub_text)])[0].flatten()
                else:
                    vector_for_key = np.zeros(self.vectors.shape[1])
                vectors_for_keys.append(vector_for_key)

            if not vectors_for_keys:
                continue

            # Average the vectors obtained for each key
            averaged_vector = sum(vectors_for_keys) / len(vectors_for_keys)

            # Update or initialize the averaged_vector for this document
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
        if abs(skip_doc) >= len(documents):
            print(f"The absolute value of skip_doc ({abs(skip_doc)}) is equal or greater than the total number of documents ({len(documents)}).")
            raise Exception("The absolute value of skip_doc is equal or greater than the total number of documents")

        if skip_doc > 0:
            kept_indices = np.arange(skip_doc, len(documents))
        elif skip_doc < 0:
            kept_indices = np.arange(0, len(documents) + skip_doc)
        else:
            kept_indices = np.arange(len(documents))

        return vectors[kept_indices], [documents[i] for i in kept_indices], kept_indices.tolist()

    def tokenize(self, text):
        """
        Tokenizes a string into words, removing punctuation.
        """
        text = ''.join(c for c in text if c not in string.punctuation)
        return set(re.findall(r'\b\w+\b', text.lower()))

    def recursive_sentence_filter(self, obj, sentence_filter_tokens):
        """
        Recursively traverse an object to find the sentence filter, considering whole words.
        The sentence_filter is tokenized only once and passed as a set of tokens.
        """
        if isinstance(obj, dict):
            return any(self.recursive_sentence_filter(v, sentence_filter_tokens) for v in obj.values())
        elif isinstance(obj, list):
            return any(self.recursive_sentence_filter(v, sentence_filter_tokens) for v in obj)
        elif isinstance(obj, str):
            # Tokenize the object string
            obj_tokens = self.tokenize(obj)
            # Check if all tokens in the sentence filter are present in the object string
            return sentence_filter_tokens.issubset(obj_tokens)
        else:
            return False

    def filter_by_sentence(self, vectors, documents, sentence_filters):
        if not isinstance(sentence_filters, (list, tuple)):
            sentence_filters = [sentence_filters]  # Ensure sentence_filters is a list or tuple

        # Tokenize each sentence filter once
        tokenized_sentence_filters = [self.tokenize(sentence_filter) for sentence_filter in sentence_filters]

        filtered_vectors = []
        filtered_documents = []

        for vec, doc in zip(vectors, documents):
            # Check if all sentence filters are found in the document
            if all(self.recursive_sentence_filter(doc, tokenized_sentence_filter) for tokenized_sentence_filter in tokenized_sentence_filters):
                filtered_vectors.append(vec)
                filtered_documents.append(doc)

        return filtered_vectors, filtered_documents

    def _is_numeric_array(self, array):
        return array.dtype.type in (np.int_, np.float_, np.intc, np.intp, np.int8,
                                    np.int16, np.int32, np.int64, np.uint8,
                                    np.uint16, np.uint32, np.uint64,
                                    np.float16, np.float32, np.float64)

    def _validate_and_reshape_array(self, array, target_shape):
        if array.ndim > 2:
            raise ValueError("query_input must be a 1D or 2D array.")
        
        # If 1D, convert to 2D
        if array.ndim == 1:
            array = np.array([array])
        
        if array.shape[1] != target_shape[1]:
            raise ValueError(f"The dimension of the query_vector ({array.shape[1]}) must match the dimension of the vectors in the database ({target_shape[1]}).")
        
        return array

    def _generate_and_validate_query_vector(self, query_input):
        try:
            if isinstance(query_input, str):
                query_vector = np.squeeze(self.generate_query_vector(query_input))
            elif isinstance(query_input, (list, np.ndarray, tuple)):
                query_input = np.array(query_input)
                if not self._is_numeric_array(query_input):
                    raise ValueError("Numeric array-like query_input expected.")
                
                query_vector = self._validate_and_reshape_array(query_input, self.vectors.shape)
            else:
                raise ValueError("query_input must be either a string or a numeric array-like object.")

            if query_vector.size == 0:
                raise ValueError("The generated query vector is empty.")
            
            return query_vector
        except Exception as e:
            print(f"An exception occurred due to invalid input: {e}")
            raise e

    def _filter_by_metadata(self, metadata_filter, filtered_vectors, filtered_documents, kept_indices=None):
        """
        A new helper method to filter vectors and documents by metadata.
        """
        # Validate the metadata keys
        self.validate_keys(metadata_filter.keys(), self.metadata_keys, "metadata_filter", "metadata_keys")

        # Create source_indices_for_filtered corresponding to filtered_documents
        source_indices_for_filtered = [self.source_indices[self.documents.index(doc)] for doc in filtered_documents]

        # Create a set of unique document indices, which will serve as the master set to filter
        unique_doc_indices = set(source_indices_for_filtered)

        # Loop through each key-value pair in the metadata filter to check for matches
        for key, value in metadata_filter.items():
            # Create an empty set to hold the indices of documents that match this key-value pair
            matching_indices = set()

            # Loop through each unique document index to find matches
            for idx in unique_doc_indices:
                metadata = self._metadata_index.get(idx, {})
                if metadata.get(key) == value:
                    matching_indices.add(idx)

            # Update the master set to be the intersection of itself and the matching_indices set
            unique_doc_indices &= matching_indices

            # If the master set is empty, break early
            if not unique_doc_indices:
                break

        # Loop through relevant_source_indices to find the full set of matching indices
        filtered_indices = [i for i, src_idx in enumerate(source_indices_for_filtered) if src_idx in unique_doc_indices]

        # Use the filtered indices to get the corresponding vectors and documents
        filtered_vectors = [filtered_vectors[i] for i in filtered_indices if i < len(filtered_vectors)]
        filtered_documents = [filtered_documents[i] for i in filtered_indices if i < len(filtered_documents)]

        return np.array(filtered_vectors, dtype=self.fp_precision), filtered_documents

    def _apply_filters(self, filters, kept_indices=None, base_vectors=None, base_documents=None):
        if base_vectors is None:
            filtered_vectors = self.vectors
        else:
            filtered_vectors = base_vectors

        if base_documents is None:
            filtered_documents = self.documents
        else:
            filtered_documents = base_documents
        
        # Initialize a set containing the IDs of all documents that haven't been filtered out
        filtered_doc_ids = set(id(doc) for doc in filtered_documents)

        # Initialize a dictionary to store filtered documents by filter name
        filtered_docs_by_filter = {}

        # Apply filters
        if filters is not None:
            for filter_name, filter_params in filters:
                if filter_name not in ['key', 'metadata', 'sentence', 'skip_doc']:
                    raise ValueError(f"Invalid filter name {filter_name}")

                if filter_name == 'skip_doc':
                    continue  # skip_doc is already applied

                # Filter by key
                if filter_name == 'key':
                    filtered_vectors, filtered_docs_by_filter['key'] = self.filter_by_key(filtered_vectors, filtered_documents, filter_params)

                # Filter by metadata
                elif filter_name == 'metadata':
                    if not self.metadata_keys:
                        raise ValueError("The 'metadata_keys' parameter has not been set in HyperDB(). Cannot filter by metadata.")
                    # Convert filter_params from tuple to dictionary
                    filter_params_dict = dict(filter_params)
                    _, filtered_docs_by_filter['metadata'] = self._filter_by_metadata(filter_params_dict, filtered_vectors, filtered_documents, kept_indices=kept_indices)

                # Filter by sentence
                elif filter_name == 'sentence':
                    _, filtered_docs_by_filter['sentence'] = self.filter_by_sentence(filtered_vectors, filtered_documents, filter_params)

                # Update the set of filtered document IDs based on this filter
                current_filtered_ids = set(id(doc) for doc in filtered_docs_by_filter[filter_name])
                filtered_doc_ids &= current_filtered_ids

        # Filter the vectors and documents based on the intersection of all filters
        filtered_vectors = [vec for vec, doc in zip(filtered_vectors, filtered_documents) if id(doc) in filtered_doc_ids]
        filtered_documents = [doc for doc in filtered_documents if id(doc) in filtered_doc_ids]

        return filtered_vectors, filtered_documents

    def _handle_timestamps(self, recency_bias, timestamp_key, filtered_documents):
        """
        Handles the computation of recency scores based on timestamps and recency bias.
        
        Args:
        - recency_bias (float): The factor by which to bias the recency of documents.
        - timestamp_key (str): The key to use for extracting timestamps from document metadata.
        - filtered_documents (list): A list of filtered documents.
        """

        # Return None if recency bias is 0 (i.e., recency is not considered)
        if recency_bias == 0:
            return None

        # Default to "timestamp" if no specific key is provided
        if timestamp_key is None:
            timestamp_key = "timestamp"

        # Check if the timestamp_key is valid
        if timestamp_key not in self.metadata_keys:
            raise ValueError(f"The timestamp_key '{timestamp_key}' must be present in metadata_keys when recency_bias is not 0.")

        # Extract timestamps from the filtered documents
        nested_keys = timestamp_key.split('.') if '.' in timestamp_key else [timestamp_key]
        timestamps = [self.get_nested_value(document, nested_keys) for document in filtered_documents]

        # Check for missing timestamps
        if None in timestamps:
            raise ValueError("All timestamps must be populated when recency_bias is not 0 or timestamp_key is provided.")

        # Convert timestamps to an array of floats
        timestamps = np.array(timestamps, dtype=float)
        
        # Compute recency scores using the exponential function
        recency_scores = recency_bias * np.exp(-np.max(timestamps) + timestamps)
        
        return recency_scores

    def _apply_ann_pre_filter(self, query_vector, ann_candidate_size, filtered_vectors, filtered_documents):
        """
        Uses ANN to get a list of candidate vectors and documents.
        The number of candidates is determined by ann_candidate_size.
        """
        if self.ann_index is None:
            raise ValueError("ANN index has not been built.")
        
        if query_vector.size != self.ann_dim:
            raise ValueError(f"Query vector dimension ({query_vector.size}) must match the Annoy index dimension ({self.ann_dim})")
      
        # Normalize the query vector if the Annoy index uses normalized vectors
        if self.vectors_normalized:
            query_vector = ranking.get_norm_vector(query_vector)
      
        ann_candidates, ann_distances = self.ann_index.get_nns_by_vector(query_vector, ann_candidate_size, include_distances=True)
        candidate_vectors = [filtered_vectors[i] for i in ann_candidates if i < len(filtered_vectors)]
        candidate_documents = [filtered_documents[i] for i in ann_candidates if i < len(filtered_documents)]
        return candidate_vectors, candidate_documents, ann_distances 

    def _hashable_key(self, query_input, top_k, return_similarities, filters, recency_bias, timestamp_key, metric, ann_percent):
        if isinstance(query_input, np.ndarray):
            query_input = tuple(query_input.tolist())
        if filters is None:
            hashable_filters = None
        else:
            # Use a generator expression to create a tuple directly without an intermediate list
            hashable_filters = tuple(
                (filter_name, tuple(sorted(filter_params.items())) if isinstance(filter_params, dict) else tuple(filter_params) if isinstance(filter_params, list) else filter_params)
                for filter_name, filter_params in filters
            )
        return (query_input, top_k, return_similarities, hashable_filters, recency_bias, timestamp_key, metric, ann_percent)

    def _cached_query(self, hashable_key):
        if hashable_key in self.lru_cache:
            self.cache_hits += 1
            return self.lru_cache[hashable_key]
        self.cache_misses += 1
        result = self._execute_query(*hashable_key)
        self.lru_cache[hashable_key] = result
        return result

    def clear_cache(self):
        """
        Clears the query cache.
        """
        self.lru_cache.clear()
        self.cache_hits = 0
        self.cache_misses = 0

    def get_cache_size_and_info(self):
        cache_info = {
            'hits': self.cache_hits,
            'misses': self.cache_misses,
            'maxsize': self.lru_cache.maxsize,
            'currsize': len(self.lru_cache)
        }
        cache_size_bytes = asizeof.asizeof(self.lru_cache)

        # Determine the appropriate unit for cache_size
        if cache_size_bytes >= (1024 * 1024):  # More than 1 MB
            cache_size_unit = 'MB'
            cache_size_value = float(cache_size_bytes) / (1024 * 1024)
        elif cache_size_bytes >= 1024:  # More than 1 KB
            cache_size_unit = 'KB'
            cache_size_value = float(cache_size_bytes) / 1024
        else:
            cache_size_unit = 'bytes'
            cache_size_value = cache_size_bytes
            
        # Format the cache size string
        if cache_size_unit == 'bytes':
            cache_size_str = f'{int(cache_size_value)} {cache_size_unit}'  # No decimal places for bytes
        else:
            cache_size_str = f'{cache_size_value:.2f} {cache_size_unit}'  # Format to 2 decimal places for KB and MB

        return {
            'cache_info': cache_info,
            'cache_memory_size': cache_size_str
        }
 
    def _execute_query(self, query_input, top_k=5, return_similarities=True, filters=None, recency_bias=0, timestamp_key=None, metric='cosine_similarity', ann_percent=5):  
        """
        Query the document store to retrieve relevant documents.
        
        Args:
        - query_input (str or array-like): The query as a string or as a vector.
        - top_k (int): Number of top matches to return.
        - return_similarities (bool): Whether to return similarity scores and indexes of the documents returned.
        - filters (list): List of tuples for filters, each tuple contains a filter name and its parameters ('key', 'metadata', 'sentence', 'skip_doc')
        - recency_bias (float): Bias toward more recent documents.
        - timestamp_key (str): Key for recency bias.
        - metric (str): Metric for similarity.
        - ann_percent (int): Percentage of total documents for ANN pre-filtering.
        """
        
        # Check if the database is empty
        if self.vectors is None or len(self.vectors) == 0 or not self.documents:
            raise Exception("The database is empty. Cannot proceed with the query.")
        
        # Validate the metric
        if metric not in ['dot_product', 'cosine_similarity', 'euclidean_metric', 'manhattan_distance', 'jaccard_similarity', 'pearson_correlation', 'hamming_distance']:
            raise ValueError(f"Invalid metric '{metric}'. Supported: 'dot_product', 'cosine_similarity', 'euclidean_metric', 'manhattan_distance', 'jaccard_similarity', 'pearson_correlation', 'hamming_distance'")
        
        # Map metrics to their ANN-compatible counterparts
        metric_mapping = {
            'dot_product': 'dot',
            'cosine_similarity': 'cosine',
            'euclidean_metric': 'euclidean',
            'manhattan_distance': 'manhattan',
            'hamming_distance': 'hamming',
        }
        
        filtered_vectors = self.vectors
        filtered_documents = self.documents
        
        try:
            query_vector = np.squeeze(self._generate_and_validate_query_vector(query_input))
            
            # Check if the metric is compatible with ANN
            ann_metric = metric_mapping.get(metric, None)
            use_ann = (ann_metric == self.ann_metric)
            
            # Apply skip_doc filter to the entire database first if specified
            kept_indices = None
            skip_active = False
            if filters:
                for filter_name, filter_params in filters:
                    if filter_name == 'skip_doc':
                        filtered_vectors, filtered_documents, kept_indices = self.apply_skip_doc(self.vectors, self.documents, filter_params)
                        skip_vectors = filtered_vectors
                        skip_documents = filtered_documents
                        skip_active = True
                        break
                        
            # If ANN is not compatible with the metric, skip the ANN pre-filtering step
            if use_ann:
                ann_candidate_size = max(top_k * 20, ((len(filtered_documents) * ann_percent) + 99) // 100)
                filtered_vectors, filtered_documents, ann_distances = self._apply_ann_pre_filter(query_vector, ann_candidate_size, filtered_vectors, filtered_documents)
                #print(f"Candidates docs: {(filtered_documents)}")
            else:
                print(f"INFO: Metric '{metric}' is not supported by the current ANN index ('{self.ann_metric}'). Bruteforce method used instead.")
            
            # Apply filters
            if filters:
                filtered_vectors, filtered_documents = self._apply_filters(filters, kept_indices=kept_indices, base_vectors=filtered_vectors, base_documents=filtered_documents)
                
            # If recency bias is applied
            if use_ann and recency_bias != 0:
                timestamps = self._handle_timestamps(recency_bias, timestamp_key, filtered_documents)
                
                # Determine if higher scores are better for the metric used
                higher_is_better = True if metric in ['dot_product', 'cosine_similarity'] else False

                # Combine ANN distances and recency scores
                # Use addition or subtraction based on whether higher or lower scores are better
                if higher_is_better:
                    combined_scores = ann_distances + timestamps
                else:
                    combined_scores = ann_distances - timestamps

                # Sort indices based on combined scores
                # Use ascending or descending sort based on whether higher or lower scores are better
                if higher_is_better:
                    sorted_indices = np.argsort(-combined_scores)[:top_k]
                else:
                    sorted_indices = np.argsort(combined_scores)[:top_k]
                
                # Final documents and scores based on the sorted indices
                final_documents = [filtered_documents[i] for i in sorted_indices]
                final_scores = combined_scores[sorted_indices]
                
                return list(zip(final_documents, final_scores)) if return_similarities else final_documents                

            # Fallback to brute-force if no results after ANN pre-filtering + filters
            if len(filtered_vectors) == 0:
                if filters:  # Only when filters are used
                    print("INFO: Falling back to brute-force search after no results from ANN pre-filtering.")
                    
                    if skip_active:
                       filtered_vectors, filtered_documents = self._apply_filters(filters, kept_indices=kept_indices, base_vectors=skip_vectors, base_documents=skip_documents)
                    else:
                        filtered_vectors, filtered_documents = self._apply_filters(filters, kept_indices=kept_indices, base_vectors=None, base_documents=None)  # Apply filters to the entire dataset
                else:
                    print("INFO: No document matches your query.")
                    return []

            # If no documents match the query after brute-force method
            if len(filtered_vectors) == 0:
                print("INFO: No document matches your query with the brute-force method and the current filters.")
                return []

            # If top_k is greater than the number of filtered documents
            if top_k > len(filtered_documents):
                print(f"Warning: top_k ({top_k}) is greater than the number of filtered documents ({len(filtered_documents)}). Setting top_k to {len(filtered_documents)}.")
                top_k = len(filtered_documents)

            # If ANN is used
            if use_ann:
                if self.ann_metric == 'cosine' and metric == 'cosine_similarity':
                    ann_distances = [1 - (d ** 2) / 2 for d in ann_distances]
                if return_similarities:
                    return [(filtered_documents[i], ann_distances[i], self.source_indices[self.documents.index(filtered_documents[i])]) for i in range(top_k)]
                else:
                    return [filtered_documents[i] for i in range(top_k)]
            
            # If ANN is not used or metric is not supported by ANN
            timestamps = self._handle_timestamps(recency_bias, timestamp_key, filtered_documents)
            ranked_results, scores = ranking.hyperDB_ranking_algorithm_sort(
                filtered_vectors, query_vector, top_k=top_k, metric=metric, timestamps=timestamps, recency_bias=recency_bias
            )
            
            # Check for invalid indices in ranked_results
            if max(ranked_results) >= len(filtered_documents):
                raise IndexError(f"Invalid index in ranked_results. Max index: {max(ranked_results)}, Length of filtered_documents: {len(filtered_documents)}")
            
            # Return the ranked documents and scores
            final_results = []
            for i, index in enumerate(ranked_results[:top_k]):
                document = filtered_documents[index]
                original_index = self.documents.index(document)  # Get the original index from the unfiltered documents
                source_index = self.source_indices[original_index]  # Get the corresponding source index
                if return_similarities:
                    final_results.append((document, scores[i], source_index))
                else:
                    final_results.append(document)

            return final_results
        
        except (ValueError, TypeError) as e:
            print(f"An exception occurred due to invalid input: {e}")
            raise e
        except Exception as e:
            print(f"An unknown exception occurred: {e}")
            raise
            
    def query(self, query_input, top_k=5, return_similarities=True, filters=None, recency_bias=0, timestamp_key=None, metric='cosine_similarity', ann_percent=5):  
        hashable_key = self._hashable_key(query_input, top_k, return_similarities, filters, recency_bias, timestamp_key, metric, ann_percent)
        return self._cached_query(hashable_key)
