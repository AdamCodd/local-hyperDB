import gzip
import pickle
import json
import sqlite3
import datetime
import numpy as np
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

def get_embedding(documents, key=None):
    """Embedding function that uses Sentence Transformers."""
    global EMBEDDING_MODEL, tokenizer
    if EMBEDDING_MODEL is None or tokenizer is None:
        # Change device to "gpu" if you want to use that instead
        EMBEDDING_MODEL = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2', device='cpu')
        tokenizer = BertTokenizer.from_pretrained('sentence-transformers/all-MiniLM-L6-v2')
    texts = []
    source_indices = []  # to track the source document of each chunk
    split_info = {}  # to track which documents were split
    if isinstance(documents, str):  # Handle case when documents is a single string
        # split long texts into chunks of max_length
        chunks = text_to_chunks(documents, tokenizer)
        if len(chunks) > 1:
            split_info[0] = True
        for chunk in chunks:
            texts.append(chunk)
            source_indices.append(0)
    elif isinstance(documents, list):
        if isinstance(documents[0], dict):
            for i, doc in enumerate(documents):  # Add enumerate here to get the index i
                if isinstance(key, str):
                    if "." in key:
                        key_chain = key.split(".")
                    else:
                        key_chain = [key]
                    for key in key_chain:
                        text = doc[key]
                        # split long texts into chunks of max_length
                        chunks = text_to_chunks(text, tokenizer)
                        if len(chunks) > 1:
                            split_info[i] = True
                        for chunk in chunks:
                            # add the key as prefix to the chunk
                            texts.append(f"{key}: {chunk}")
                            source_indices.append(i)
                elif key is None:
                    for key, value in doc.items():
                        # split long texts into chunks of max_length
                        chunks = text_to_chunks(value, tokenizer)
                        if len(chunks) > 1:
                            split_info[i] = True
                        for chunk in chunks:
                            # add the key as prefix to the chunk
                            texts.append(f"{key}: {chunk}")
                            source_indices.append(i)
        elif isinstance(documents[0], str):
            texts = documents
            source_indices = list(range(len(documents)))
    embeddings = EMBEDDING_MODEL.encode(texts).astype(np.float16)
    return embeddings, source_indices, split_info


class HyperDB:
    def __init__(
        self,
        documents=None,
        vectors=None,
        key=None,
        embedding_function=None,
        similarity_metric="cosine",
    ):
        self.source_indices = []
        self.split_info = {}
        documents = documents or []
        self.documents = []
        self.pending_vectors = []  # Store vectors that need to be stacked
        self.vectors = None
        self.embedding_function = embedding_function or (
            lambda docs: get_embedding(docs, key=key)
        )
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

    def dict(self, vectors=False):
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

    def add(self, documents, vectors=None):
        if not isinstance(documents, list):
            return self.add_document(documents, vectors)
        self.add_documents(documents, vectors)

    def add_document(self, document: dict, vectors=None, count=1):
        if vectors is None:
            vectors, _, _ = self.embedding_function([document])
        if self.vectors is None:
            self.vectors = np.empty((0, vectors[0].shape[0]), dtype=np.float16)
        
        vector_list = []  # Temporary list to store new vectors
        
        for vector in vectors:
            if len(vector) != self.vectors.shape[1]:
                print(f"Dimension mismatch. Got vector of dimension {len(vector)} while the existing vectors are of dimension {self.vectors.shape[1]}")
                return  
            vector_list.append(vector.astype(np.float16))
            
        # Add all vectors from the vector_list to self.vectors
        self.vectors = np.vstack([self.vectors, *vector_list])
        
        timestamp = datetime.datetime.now().timestamp()
        document['timestamp'] = str(timestamp)
        self.documents.extend([document]*count)  # Extend the document list with the same document for all chunks
        self.source_indices.extend([self.documents.index(document)]*count)  # Extend the source_indices list with the same index for all chunks

    def add_documents(self, documents, vectors=None):
        if not documents:
            return
        if vectors is None:
            vectors, source_indices, split_info = self.embedding_function(documents)
        else:
            source_indices = list(range(len(documents)))
        
        self.source_indices.extend(source_indices)  # store the source indices
        
        for i, document in enumerate(documents):
            count = split_info.get(i, 1)  # Get the number of chunks for this document
            self.add_document(document, vectors[i], count)  # Provide the list of vectors to add_document
    
    def remove_document(self, indices):
        # Ensure indices is a list
        if isinstance(indices, int):
            indices = [indices]

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

            self.vectors = np.array(data["vectors"], dtype=np.float16)
            self.documents = data["documents"]
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

    def query(self, query_text, top_k=5, return_similarities=True, recency_bias=0.2):
        try:
            query_vector = self.embedding_function([query_text])[0]
            # Adding a timestamp to each document
            timestamps = [document['timestamp'] for document in self.documents]
            ranked_results, combined_scores, original_similarities = custom_ranking_algorithm_sort(
                self.vectors, query_vector, timestamps, top_k=top_k, metric=self.similarity_metric, recency_bias=recency_bias
            )
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
