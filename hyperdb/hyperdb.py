import gzip
import pickle
import datetime
import numpy as np

from transformers import BertTokenizer
from fast_sentence_transformers import FastSentenceTransformer as SentenceTransformer

from hyperdb.galaxy_brain_math_shit import (
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

# Maximum max_length=256 for all-MiniLM-L6-v2
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
                        # Split long documents into chunks of max_length
                        chunks = text_to_chunks(text, tokenizer)
                        if len(chunks) > 1:
                            split_info[i] = True
                        for chunk in chunks:
                            # Add the key as prefix to the chunk
                            texts.append(f"{key}: {chunk}")
                            source_indices.append(i)
                elif key is None:
                    for key, value in doc.items():
                        # Split long documents into chunks of max_length
                        chunks = text_to_chunks(value, tokenizer)
                        if len(chunks) > 1:
                            split_info[i] = True
                        for chunk in chunks:
                            # Add the key as prefix to the chunk
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
        for vector in vectors:
            if len(vector) != self.vectors.shape[1]:
                print(f"Dimension mismatch. Got vector of dimension {len(vector)} while the existing vectors are of dimension {self.vectors.shape[1]}")
                return  
            self.vectors = np.vstack([self.vectors, vector.astype(np.float16)])      
        timestamp = datetime.datetime.now().timestamp()
        document['timestamp'] = timestamp
        self.documents.extend([document]*count)  # Extend the document list with the same document for all chunks
        self.source_indices.extend([self.documents.index(document)]*count)  # Extend the source_indices list with the same index for all chunks

    def add_documents(self, documents, vectors=None):
        if not documents:
            return
        if vectors is None:
            vectors, source_indices, split_info = self.embedding_function(documents)
        else:
            source_indices = list(range(len(documents)))
        self.source_indices.extend(source_indices)  # Store the source indices
        for i, document in enumerate(documents):
            count = split_info.get(i, 1)  # Get the number of chunks for this document
            self.add_document(document, vectors[i], count)  # Provide the list of vectors to add_document

    def remove_document(self, index):
        self.vectors = np.delete(self.vectors, index, axis=0)
        self.documents.pop(index)
      
    def save(self, storage_file):
        data = {"vectors": self.vectors, "documents": self.documents}
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "wb") as f:
                pickle.dump(data, f)
        else:
            with open(storage_file, "wb") as f:
                pickle.dump(data, f)

    def load(self, storage_file):
        if storage_file.endswith(".gz"):
            with gzip.open(storage_file, "rb") as f:
                data = pickle.load(f)
        else:
            with open(storage_file, "rb") as f:
                data = pickle.load(f)
        self.vectors = data["vectors"].astype(np.float16)
        self.documents = data["documents"]

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
        except Exception as e:
            print(f"An exception occurred: {e}")
