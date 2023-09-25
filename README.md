# HyperDB
<div>
<img src="https://github.com/jdagdelen/hyperDB/blob/main/_static/logo.png?raw=true" width="400" alt="HyperDB Logo">
</div>

A hyper-fast local vector database for use with LLM Agents. Highly optimized C++ backend vector store with HW accelerated operations via MKL BLAS and enables users to use their own embedding function or let HyperDB embed the documents.

## Forked from [jdagdelen](https://github.com/jdagdelen/hyperDB)
This fork significantly extends the original Vector Database project, removing all OpenAI dependencies to run fully locally using SentenceTransformer. We've introduced several features and optimizations to enhance performance, flexibility, and the user experience.

## Major changes:
### Performance and Scalability
* <b>Token-based Chunking</b>: Implements a technique to handle embeddings of documents that exceed the model's 256-token limit.
* <b>Data Types</b>: Extends support for vector data types to include FP16, FP32, and FP64 (default: FP32).
* <b>Batch Operations</b>: Streamlines batch insertion and deletion of documents for enhanced efficiency.

### Query Enhancements
* <b>Selective Inclusion</b>: Adds `skip_doc` parameter to the `query` method to selectively include documents before ranking, focusing search results.
* <b>Time-Decay Ranking</b>: Introduces a custom ranking algorithm incorporating a time-decay factor for recency bias.
* <b>Vector-Based Queries</b>: Incorporates `query_vector` parameter in the `query` method for direct vector-based queries alongside traditional text queries.
* <b>Dynamic Metric Selection</b>: Extends the query method to allow the selection of similarity metrics, including Hamming distance, dot product, and Euclidean metric and more, for more tailored search results.

### Data Storage and Retrieval
* <b>Storage Formats</b>: Extends data storage compatibility to include JSON and SQLite formats, in addition to Pickle.
* <b>Timestamp Support</b>: Enables optional timestamping of individual documents with a configurable key for query optimization.

### Analytics and Testing
* <b>Ranking Algorithm Tests</b>: Enhances the robustness of ranking algorithm tests for improved accuracy.
* <b>Word Frequency Analysis</b>: Integrates an optional feature for in-depth database analytics based on word frequency.

### Advanced Filtering and Targeting
* <b>Key-Based Filtering</b>: Enables advanced key-based filtering of documents prior to embedding. Supports multiple and nested keys for enhanced model flexibility and targeting.
* <b>Vector Similarity Research</b>: Introduces the capability to compare the similarity between vectors, such as document embeddings, for clustering, categorization, or anomaly detection.
* <b>Sentence-Level Similarity Search</b>: Extends the key-driven similarity search to the granularity of individual sentences within documents. This allows for a more nuanced understanding of document content and enables more precise retrieval of relevant information.

### Filtering order
1. <b>Key-Based Filter</b>: If your documents are already categorized or tagged with certain keys, the `key` filter could significantly reduce the computational load for subsequent steps.
2. <b>Sentence-Level Filter</b>: After narrowing down the list of documents with a key, you may still have a large number of documents to sift through. The `sentence_filter` can help further narrow this down based on content relevance.
3. <b>Skip-Doc Filter</b>: After all other filtering mechanisms have been applied, the `skip_doc` parameter could be used to skip the first N or last N documents for final ranking. This would be like a "fine-tuning" step.
These filters are optional and can be used either individually or in combination to improve search results and performance. 
 
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

# Load Pokémon data from a JSONL file into a list of dictionaries
documents = []
with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Create a HyperDB instance and index the Pokémon descriptions
db = HyperDB(documents)

# Save the database to a file
db.save("demo/pokemon_hyperdb.pickle.gz")

# Load the database from the file
db.load("demo/pokemon_hyperdb.pickle.gz")

# Perform a query to find Pokémon that like to sleep
results = db.query("Likes to sleep.", top_k=3)
```
Formatting the results:
```python
# Helper function to print list items
def print_list(lst, indent=0):
    for i, item in enumerate(lst):
        if isinstance(item, dict):
            item_str = ", ".join([f"{k}={v}" for k, v in item.items()])
            print("  " * indent + f"{i + 1}. {item_str}")
        else:
            print("  " * indent + f"{i + 1}. {item}")

# Helper function to print dictionary items
def print_dict(d, indent=0):
    for key, value in d.items():
        if isinstance(value, dict):
            print("  " * indent + f"{key.capitalize()}:")
            print_dict(value, indent + 1)
        elif isinstance(value, list):
            print("  " * indent + f"{key.capitalize()}:")
            print_list(value, indent + 1)
        else:
            print("  " * indent + f"{key.capitalize()}: {value}")

# Function to print query results
def print_pokemon_info(results):
    for res in results:
        document, similarity = res  # Unpack result tuple
        print_dict(document)  # Pretty-print the Pokémon data
        print(f"Similarity: {similarity}") # Optional - print the similarity score
        print("-" * 40)  # Add a separator between entries

# Display the query results
print_pokemon_info(results)
```

Returns:
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
```

### Partial document embedding through key-based selection:

```python
# Instantiate a HyperDB instance with a list of documents and specify the key as "name" for embedding generation.
# The instance will focus solely on the 'name' field within each document to create the embeddings.
# The `key` parameter also supports multiple keys and nested keys for more complex filtering prior to embedding.
db = HyperDB(documents, key="name")

# Save the HyperDB instance to a file
db.save(f"testing\pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the save file
db.load(f"testing\pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance with a text input
results = db.query("Pika", top_k=3)
```
Returns:
```
Name: Pikachu
----------------------------------------
Name: Pidgeot
----------------------------------------
Name: Pidgey
----------------------------------------
```

### Partial document querying through key-based selection:
```python
# Instantiate HyperDB
db = HyperDB(documents)

# Save the HyperDB instance to a file
db.save(f"testing\pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the save file
db.load(f"testing\pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance using a text input ("Pikachu") and specify the key as "info.description" for filtering documents.
# The `key` parameter also supports multiple keys and nested keys for more complex filtering.
results = db.query("Pikachu", top_k=3, key="info.description")
```
Returns:
```
Name: Pikachu
Shortname: pikachu
Hp: 160
Info:
  Id: 25
  Type: electric
  Weakness: ground
  Description: Melissa's favorite Pokemon! When several Pikachu gather, their electricity could build and cause lightning storms.
Images:
  Photo: images/pikachu.jpg
  Typeicon: icons/electric.jpg
  Weaknessicon: icons/ground.jpg
Moves:
  1. name=Growl, type=normal
  2. name=Quick Attack, dp=40, type=normal
  3. name=Thunderbolt, dp=90, type=electric
----------------------------------------
Name: Porygon
Shortname: porygon
Hp: 160
Info:
  Id: 137
  Type: normal
  Weakness: fighting
  Description: A manmade Pokemon that came about as a result of research. It is programmed with only basic motions.
Images:
  Photo: images/porygon.jpg
  Typeicon: icons/normal.jpg
  Weaknessicon: icons/fighting.jpg
Moves:
  1. name=Psybeam, dp=65, type=psychic
  2. name=Sharpen, type=normal
  3. name=Tri Attack, dp=80, type=normal
----------------------------------------
Name: Arcanine
Shortname: arcanine
Hp: 290
Info:
  Id: 59
  Type: fire
  Weakness: water
  Description: This legendary Chinese Pokemon is considered magnificent. Many people are enchanted by its grand mane.
Images:
  Photo: images/arcanine.jpg
  Typeicon: icons/fire.jpg
  Weaknessicon: icons/water.jpg
Moves:
  1. name=Bite, dp=60, type=dark
  2. name=Double Team, type=normal
  3. name=Extreme Speed, dp=80, type=normal
  4. name=Fire Fang, dp=65, type=fire
----------------------------------------
Moves:
  1. name=Bite, dp=60, type=dark
  2. name=Tackle, dp=60, type=normal
  3. name=Water Gun, dp=40, type=water
  4. name=Withdraw, type=water
```
### Partial document querying through skip_doc parameter:
The `skip_doc` parameter allows you to selectively include or exclude a certain number of documents before applying the ranking algorithm in the query method. If `skip_doc` is a positive integer, the method will skip the first `skip_doc` number of documents. If it is a negative integer, the method will exclude the last `skip_doc` number of documents.

Note: If the absolute value of `skip_doc` is greater than the total number of documents, a warning will be shown.

Example:
```python
# Initialize HyperDB
db = HyperDB()

# Add some documents to the database
db.add(["Document 1", "Document 2", "Document 3", "Document 4", "Document 5"])

# Query with skip_doc = 2, this will skip the first two documents before ranking
# Only "Document 3", "Document 4", and "Document 5" would be considered for ranking and the top 2 among them will be returned.
result_1 = db.query("Some query text", top_k=2, skip_doc=2)

# Query with skip_doc = -2, this will exclude the last two documents before ranking
# Only "Document 1", "Document 2", and "Document 3" would be considered for ranking and the top 2 among them will be returned.
result_2 = db.query("Some query text", top_k=2, skip_doc=-2)

# Query with skip_doc = 0 (default), this will include all documents in ranking
# All documents would be considered for ranking and the top 2 among them will be returned.
result_3 = db.query("Some query text", top_k=2)
```
