# HyperDB
<div>
<img src="https://github.com/jdagdelen/hyperDB/blob/main/_static/logo.png?raw=true" width="400" alt="HyperDB Logo">
</div>

A hyper-fast local vector database for use with LLM Agents.

## Forked from [jdagdelen](https://github.com/jdagdelen/hyperDB)
This fork removes all OpenAI requirements making this vector database running fully local (using SentenceTransformer).

Major changes:
* Handles embeddings of long documents (exceeding the 256 tokens of the model limitation) by splitting them into chunks
* Handles single long string as documents
* Vectors are saved into FP16 instead of FP32 to improve the speed and reduce the size of the vector database
* Each document is timestamped
* A custom ranking algorithm has been added which add a recency bias (optional) to documents while computing the similarity scores
* Add/remove documents in batch efficently

## Advantages
* Simple interface compatible with _all_ large language model agents. 
* Highly optimized C++ backend vector store with HW accelerated operations via MKL BLAS. 
* Enables users to index documents with advanced features such as _ids_ and _metadata_.

## Installation

Install the package from PyPI:

```bash
git clone https://github.com/AdamCodd/local-hyperDB.git
cd local-hyperDB
pip install .
```

## Usage

Here's an example of using HyperDB to store and query documents with information about all 151 original pokemon _in an instant_:

```python
import json
from hyperdb import HyperDB

# Load documents from the JSONL file
documents = []

with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Instantiate HyperDB with the list of documents
db = HyperDB(documents, key="info.description")

# Save the HyperDB instance to a file
db.save("demo/pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the save file
db.load("demo/pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance with a text input
results = db.query("Likes to sleep.", top_k=5)
```

Returns 
```
Name: Snorlax
Pokedex ID: 143
HP: 160
Type: normal
Weakness: fighting
Description: Very lazy. Just eats and sleeps. As its rotund bulk builds, it becomes steadily more slothful.

Name: Drowzee
Pokedex ID: 96
HP: 230
Type: psychic
Weakness: dark
Description: Puts enemies to sleep then eats their dreams. Occasionally gets sick from eating bad dreams.

Name: Pinsir
Pokedex ID: 127
HP: 160
Type: bug
Weakness: fire
Description: When the temperature drops at night, it sleeps on treetops or among roots where it is well hidden.

Name: Abra
Pokedex ID: 63
HP: 160
Type: psychic
Weakness: dark
Description: Sleeps 18 hours a day. If it senses danger, it will teleport itself to safety even as it sleeps.

Name: Venonat
Pokedex ID: 48
HP: 160
Type: bug
Weakness: fire
Description: Lives in the shadows of tall trees where it eats insects. It is attracted by light at night.
```

<img width="600" src="https://raw.githubusercontent.com/jdagdelen/hyperDB/main/_static/0B147C7D-BEB0-4E61-9397-64A460C8CE22.png"/>

*Benchmark Credit: Benim Kıçım
