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
* Each document can be timestamped (optional) and a custom key can be chosen as timestamp to query the vector database
* A custom ranking algorithm has been added which add a recency bias (optional) to documents while computing the similarity scores
* Add/remove documents in batch efficently
* JSON/Sqlite support along with pickle
* Improved testing of ranking algorithms
* Added a word frequency capability to HyperDB for analysis purposes (optional) 

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

# Print the query results in a human-readable format
def print_pokemon_info(results):
    def print_dict_items(d, indent=0):
        for key, value in d.items():
            if isinstance(value, dict):
                print("  " * indent + f"{key.capitalize()}:")
                print_dict_items(value, indent + 1)
            elif isinstance(value, list):
                print("  " * indent + f"{key.capitalize()}:")
                for i, item in enumerate(value):
                    if isinstance(item, dict):
                        item_str = ", ".join([f"{k}={v}" for k, v in item.items()])
                        print("  " * (indent + 1) + f"{i + 1}. {item_str}")
                    else:
                        print("  " * (indent + 1) + f"{i + 1}. {item}")
            else:
                print("  " * indent + f"{key.capitalize()}: {value}")
                
    # Iterate through the query results
    for res in results:
        document, score1, score2 = res  # Unpacking the tuple
        print_dict_items(document)
        print("-" * 40)  # Separator
print_pokemon_info(results)
```

Returns 
```
Name: Drowzee
Shortname: drowzee
Hp: 230
Info:
  Id: 96
  Type: psychic
  Weakness: dark
  Description: Puts enemies to sleep then eats their dreams. Occasionally gets sick from eating bad dreams.
Images:
  Photo: images/drowzee.jpg
  Typeicon: icons/psychic.jpg
  Weaknessicon: icons/dark.jpg
Moves:
  1. name=Headbutt, dp=70, type=normal
  2. name=Ice Punch, dp=75, type=ice
  3. name=Meditate, type=psychic
  4. name=Psybeam, dp=65, type=psychic
----------------------------------------
Name: Snorlax
Shortname: snorlax
Hp: 160
Info:
  Id: 143
  Type: normal
  Weakness: fighting
  Description: Very lazy. Just eats and sleeps. As its rotund bulk builds, it becomes steadily more slothful.
Images:
  Photo: images/snorlax.jpg
  Typeicon: icons/normal.jpg
  Weaknessicon: icons/fighting.jpg
Moves:
  1. name=Amnesia, type=psychic
----------------------------------------
Name: Jigglypuff
Shortname: jigglypuff
Hp: 340
Info:
  Id: 39
  Type: fairy
  Weakness: poison
  Description: Garrett's favorite pokemon! When its huge eyes light up, it sings a mysteriously soothing melody that lulls its enemies to sleep.
Images:
  Photo: images/jigglypuff.jpg
  Typeicon: icons/fairy.jpg
  Weaknessicon: icons/poison.jpg
Moves:
  1. name=Defense Curl, type=normal
  2. name=Pound, dp=40, type=normal
  3. name=Rollout, dp=30, type=rock
  4. name=Wakeup Slap, dp=70, type=fighting
----------------------------------------
Name: Abra
Shortname: abra
Hp: 160
Info:
  Id: 63
  Type: psychic
  Weakness: dark
  Description: Sleeps 18 hours a day. If it senses danger, it will teleport itself to safety even as it sleeps.
Images:
  Photo: images/abra.jpg
  Typeicon: icons/psychic.jpg
  Weaknessicon: icons/dark.jpg
Moves:
  1. name=Double Team, type=normal
  2. name=Energy Ball, dp=90, type=grass
  3. name=Psychic, dp=90, type=psychic
  4. name=Thief, dp=60, type=dark
----------------------------------------
Name: Pinsir
Shortname: pinsir
Hp: 160
Info:
  Id: 127
  Type: bug
  Weakness: fire
  Description: When the temperature drops at night, it sleeps on treetops or among roots where it is well hidden.
Images:
  Photo: images/pinsir.jpg
  Typeicon: icons/bug.jpg
  Weaknessicon: icons/fire.jpg
Moves:
  1. name=Harden, type=normal
  2. name=Vice Grip, dp=55, type=normal
----------------------------------------
```
