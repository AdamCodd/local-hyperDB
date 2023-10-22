# HyperDB
<div>
<img src="https://github.com/jdagdelen/hyperDB/blob/main/_static/logo.png?raw=true" width="400" alt="HyperDB Logo">
</div>

A fast local vector database for use with LLM Agents with extensive filters and metrics.

## Forked from [jdagdelen](https://github.com/jdagdelen/hyperDB)
This fork significantly extends the original Vector Database project, removing all OpenAI dependencies to run fully locally using SentenceTransformer. We've introduced several features and optimizations to enhance performance, flexibility, and the user experience.

## Major changes:
### Performance and Scalability
* <b>Token-based Chunking</b>: Handles embeddings of documents that exceed the model's 512 tokens limit.
* <b>Data Types</b>: Extends support for vector data types to include FP16, FP32, and FP64 (default: FP32).
* <b>Batch Operations</b>: Streamlines batch insertion and deletion of documents for enhanced efficiency.
* <b>ANN Index</b>: Construct an ANN index (using Annoy library) to expedite query processing. Supports various metrics: "angular", "euclidean", "manhattan", "hamming", "dot", "cosine" (this one was implemented manually and is not part of the library, it is the default metric) and `n_trees` to set the number of trees to be used.

### Data Storage and Retrieval
* <b>Storage Formats</b>: Extends data storage compatibility to include JSON and SQLite formats, in addition to Pickle.
* <b>Partial document storage</b>: Divide the document to embed and incorporate exclusively the key(s) designated through the `select_keys` parameter.
* <b>Metadata Support</b>: Enables the storage of additional metadata keys from each document for more granular filtering and retrieval.
* <b>Timestamp Support</b>: Enables optional timestamping of individual documents with a configurable key for query optimization.
* <b>Vector Similarity Research</b>: Introduces the capability to compare the similarity between vectors, such as document embeddings, for clustering, categorization, or anomaly detection.

### Analytics and Testing
* <b>Ranking Algorithm Tests</b>: Enhances the robustness of ranking algorithm tests for improved accuracy.
* <b>Word Frequency Analysis</b>: Integrates an optional feature for in-depth database analytics based on word frequency.

### Query Enhancements
* <b>ANN prefilter</b>: Queries are pre-filtered using ANN to accelerate subsequent filtering, with a fallback to brute-force searching in case no results are returned.
* <b>Time-Decay Ranking</b>: Incorporates a recency bias in the ranking algorithm, allowing more recent documents to be ranked higher based on a configurable time-decay factor.
* <b>Vector-Based Queries</b>: Incorporates `query_vector` parameter in the `query` method for direct vector-based queries alongside traditional text queries.
* <b>Dynamic Metric Selection</b>: Extends the query method to allow the selection of similarity metrics, including Hamming distance, dot product, and Euclidean metric and more, for more tailored search results.
* <b>Query cache</b>: Utilizes a cache for query vectors generated from string inputs, enhancing query performance for repeated queries.

### Advanced Filtering and Targeting
The `filters` parameter in the query method offers a flexible, user-defined sequence of filtering steps, allowing users to apply a combination of key-based, metadata-based, sentence-level, and document-skipping filters in any order to refine search results.
* <b>Key-Based Filtering</b>: The `key` parameter allows for the targeting of specific attributes within documents for similarity comparison. Supports multiple and nested keys for enhanced model flexibility and targeting.
* <b>Metadata-Level Filter</b>: The `metadata` parameter allows for selective inclusion of documents that match specific metadata key-value pairs.
* <b>Sentence-Level Filter</b>: The `sentence_filter` parameter can narrow down document candidates based on sentence-level content relevance.
* <b>Skip-Doc Filter</b>: The `skip_doc` parameter can be used to selectively include or exclude documents.

These filters are optional and can be used either individually or in combination to improve search results and performance. 

Keep in mind that key-based and sentence-level filters can introduce computational overhead due to the generation of new vectors and text tokenization, especially in large databases. If performance is a concern, consider using metadata-based filtering, which is designed to be more efficient as it leverages a pre-indexed metadata dictionary.
 
## Installation

Install the package from PyPI:

```bash
git clone https://github.com/AdamCodd/local-hyperDB.git
cd local-hyperDB
pip install .
```

## Usage

Here's a basic example of using HyperDB to store and query documents with information about all 151 original pokemon _in an instant_:

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
def format_entry(pokemon, score=None):
    def nested_dict_to_str(d, indent=0):
        lines = []
        for key, value in d.items():
            if isinstance(value, dict):
                lines.append("  " * indent + f"{key.capitalize()}:")
                lines.append(nested_dict_to_str(value, indent + 1))
            elif isinstance(value, list):
                lines.append("  " * indent + f"{key.capitalize()}:")
                for i, item in enumerate(value, 1):
                    if isinstance(item, dict):
                        item_str = ", ".join([f"{k}={v}" for k, v in item.items()])
                        lines.append("  " * (indent + 1) + f"{i}. {item_str}")
                    else:
                        lines.append("  " * (indent + 1) + f"{i}. {item}")
            else:
                lines.append("  " * indent + f"{key.capitalize()}: {value}")
        return "\n".join(lines)

    prettify_pokemon = nested_dict_to_str(pokemon)
    
    if score is not None:
        prettify_pokemon += f"\nSimilarity: {score}"

    return prettify_pokemon

# Function to print query results
def print_pokemon_info(results):
    for res in results:
        if len(res) == 2:
            document, similarity = res
        elif len(res) == 1:
            document = res
            similarity = None
        else:
            print(f"{len(res)} - Res: {results}")
            print("Invalid result format.")
            continue
        print(format_entry(document, similarity))  # Pretty-print the Pokémon data
        print("-" * 40)  # Add a separator between entries

# Display the query results
print_pokemon_info(results)
```

Returns:
```
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
Similarity: 0.3639167563608723
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
Similarity: 0.3239926281894441
----------------------------------------
Name: Dodrio
Shortname: dodrio
Hp: 230
Info:
  Id: 85
  Type: flying
  Weakness: electric
  Description: Uses its three brains to execute complex plans. While two heads sleep, one head stays awake.
Images:
  Photo: images/dodrio.jpg
  Typeicon: icons/flying.jpg
  Weaknessicon: icons/electric.jpg
Moves:
  1. name=Drill Peck, dp=80, type=flying
  2. name=Pursuit, dp=40, type=dark
  3. name=Swords Dance, type=normal
  4. name=Tri Attack, dp=80, type=normal
Similarity: 0.321253797550348
```

### Partial document embedding through key-based selection:
The `select_keys` parameter is flexible; it can accept multiple keys, including nested keys. For instance, specifying ['name', 'info.description'] will include both the 'name' key and the 'description' key nested under 'info'. This feature allows for more refined filtering of data prior to the embedding process.
```python
# In this HyperDB instance, the documents (formatted as dictionaries) will be processed such that only the key 'name' is retained for creating embeddings.
db = HyperDB(documents, select_keys="name")

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
When we query by `key` (using the `filters` parameter) the similarity score computation is concentrated solely on the portion of the document corresponding to that key. This can be particularly useful when the documents have multiple keys and you're interested in finding similarities based on a specific aspect of the documents.
```python
# Instantiate HyperDB
db = HyperDB(documents)

# Save the HyperDB instance to a file
db.save(f"testing\pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the save file
db.load(f"testing\pokemon_hyperdb.pickle.gz")

# The `filters` parameter supports multiple types of filters including key-based filtering.
# Query the HyperDB instance using a text input ("Pikachu") and specify the key as "info.description" to focus the similarity score on that part.
results = db.query("Pikachu", top_k=3, filters=[('key', 'info.description')])
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
Similarity: 0.572265625
----------------------------------------
Name: Cubone
Shortname: cubone
Hp: 210
Info:
  Id: 104
  Type: ground
  Weakness: water
  Description: Jack's favorite Pokemon! Cubone's both cute and completely hardcore.
Images:
  Photo: images/cubone.jpg
  Typeicon: icons/ground.jpg
  Weaknessicon: icons/water.jpg
Moves:
  1. name=Bone Club, dp=65, type=ground
  2. name=Growl, type=normal
  3. name=Headbutt, dp=70, type=normal
  4. name=Stomping Tantrum, dp=75, type=ground
Similarity: 0.449462890625
----------------------------------------
Name: Lapras
Shortname: lapras
Hp: 370
Info:
  Id: 131
  Type: water
  Weakness: electric
  Description: Nicole's favorite Pokemon! Its high intelligence enables it to understand human speech. It likes to ferry people on its back.
Images:
  Photo: images/lapras.jpg
  Typeicon: icons/water.jpg
  Weaknessicon: icons/electric.jpg
Moves:
  1. name=Growl, type=normal
  2. name=Hydro Pump, dp=110, type=water
  3. name=Thunder, dp=110, type=electric
  4. name=Ice Beam, dp=90, type=ice
Similarity: 0.430419921875
```

### Partial document querying through sentence-based selection:
When we query by `sentence` (using the `filters` parameter) we filter the documents based on the presence of a specific sentence before they are ranked through the similarity score, so only the documents that contains that sentence will be output. This process is case-insensitive and all punctuation is removed (i.e "Melissa", "melissa", "Melissa?", "Melissa!" will give the same result).
```python
# Instantiate HyperDB
db = HyperDB(documents)

# Save the HyperDB instance to a file
db.save(f"testing\pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the save file
db.load(f"testing\pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance using a vague input ("electric") and filter the documents that contains the word "Melissa".
# The `filters` parameter supports multiple types of filters, including sentence-based filtering. This type of filter will narrow down the documents considered for similarity ranking to those that contain the specified word or substring in their content.
results = db.query("electric", top_k=3, filters=[('sentence', 'Melissa')])
```

Returns:
```
Warning: top_k (3) is greater than the number of filtered documents (1). Setting top_k to 1.
Info: Only one document left.
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
Similarity: [0.49121094]
```

### Partial document querying through metadata parameter:
When you query with the `metadata` filter (using the `filters` parameter), you're asking the system to first narrow down the list of documents based on specific details in their metadata. Only the documents that "pass" this metadata filter will then be ranked by how well they match your query.
Example:

Let's say your document store has metadata fields like author and category. You can easily find all the science papers by 'John Doe' by using a metadata filter like this: <b>('metadata', ({ 'author': 'John Doe', 'category': 'Science' })</b>.

👉 Important: Make sure the keys you use in the metadata filter (author, category, etc.) are listed in `metadata_keys` when you create or load your HyperDB instance. Otherwise, the filter will throw an error.
```python
# Create HyperDB instance, specify 'info.weakness' as a metadata key
db = HyperDB(documents, metadata_keys=['info.weakness'])

# Save and then reload the HyperDB instance
db.save("testing/pokemon_hyperdb.pickle.gz")
db.load("testing/pokemon_hyperdb.pickle.gz")

# Query with metadata filter: Find documents where 'info.weakness' is 'dark'
results = db.query("pokemon", top_k=3, filters=[('metadata', {'info.weakness': 'dark'})])
```

Returns:
```
Name: Mew
Shortname: mew
Hp: 160
Info:
  Id: 151
  Type: psychic
  Weakness: dark
  Description: Its DNA is said to contain the genetic codes of all Pokemon.
Images:
  Photo: images/mew.jpg
  Typeicon: icons/psychic.jpg
  Weaknessicon: icons/dark.jpg
Moves:
  1. name=Amnesia, type=psychic
  2. name=Psychic, dp=90, type=psychic
Similarity: 0.498779296875
----------------------------------------
Name: Gengar
Shortname: gengar
Hp: 230
Info:
  Id: 94
  Type: ghost
  Weakness: dark
  Description: Chadi's favorite Pokemon! It is said to emerge from darkness to steal the lives of those who become lost in mountains.
Images:
  Photo: images/gengar.jpg
  Typeicon: icons/ghost.jpg
  Weaknessicon: icons/dark.jpg
Moves:
  1. name=Dark Pulse, dp=80, type=dark
  2. name=Double Team, type=normal
  3. name=Shadow Ball, dp=80, type=ghost
  4. name=Venoshock, dp=65, type=poison
Similarity: 0.421875
----------------------------------------
Name: Alakazam
Shortname: alakazam
Hp: 220
Info:
  Id: 65
  Type: psychic
  Weakness: dark
  Description: Its brain can outperform a supercomputer. Its intelligence quotient is said to be 5,000.
Images:
  Photo: images/alakazam.jpg
  Typeicon: icons/psychic.jpg
  Weaknessicon: icons/dark.jpg
Moves:
  1. name=Focus Blast, dp=120, type=fighting
  2. name=Kinesis, type=psychic
  3. name=Psychic, dp=90, type=psychic
  4. name=Shadow Ball, dp=80, type=ghost
Similarity: 0.33349609375
----------------------------------------
```

### Partial document querying through skip_doc parameter:
The `skip_doc` parameter allows you to selectively include or exclude a certain number of documents before applying the ranking algorithm in the query method. If `skip_doc` is a positive integer, the method will skip the first `skip_doc` number of documents. If it is a negative integer, the method will exclude the last `skip_doc` number of documents.

Note: If the absolute value of `skip_doc` is greater than the total number of documents, a warning will be shown and the filter will be ignored.

Example:
```python
# Initialize HyperDB
db = HyperDB()

# Add some documents to the database
db.add(["Document 1", "Document 2", "Document 3", "Document 4", "Document 5"])

# Query with skip_doc = 2, this will skip the first two documents before ranking
# Only "Document 3", "Document 4", and "Document 5" would be considered for ranking, and the top 2 among them will be returned.
result_1 = db.query("Some query text", top_k=2, filters=[('skip_doc', 2)])

# Query with skip_doc = -2, this will exclude the last two documents before ranking
# Only "Document 1", "Document 2", and "Document 3" would be considered for ranking, and the top 2 among them will be returned.
result_2 = db.query("Some query text", top_k=2, filters=[('skip_doc', -2)])

# Query with skip_doc = 0 (default), this will include all documents in ranking
# All documents would be considered for ranking, and the top 2 among them will be returned.
result_3 = db.query("Some query text", top_k=2)
```

### Combining Multiple Filters for Advanced Querying:
The `filters` parameter in the query method supports combining multiple filters in any order to narrow down your document set before applying the ranking algorithm. You can pass a list of tuples, where each tuple contains a filter type and its corresponding parameters.

Example:
```python
# Initialize HyperDB with metadata keys
db = HyperDB(documents, metadata_keys=['info.weakness'])

# Save the HyperDB instance to a file
db.save(f"testing\pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the save file
db.load(f"testing\pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance using a general input ("pokemon"). 
# 1) First, filter the documents that have the value "dark" for the nested key "info.weakness" in their metadata.
# 2) Then, narrow it down to those that contain the word "favorite" in their content.
# 3) Finally, focus the similarity ranking on the part of the document specified by the key "info.description".
results = db.query("pokemon", top_k=3, filters=[
    ('metadata', {'info.weakness': 'dark'}), 
    ('sentence', 'favorite'), 
    ('key', 'info.description')
])
```

Returns:
```
Warning: top_k (3) is greater than the number of filtered documents (1). Setting top_k to 1.
Info: Only one document left.
Name: Gengar
Shortname: gengar
Hp: 230
Info:
  Id: 94
  Type: ghost
  Weakness: dark
  Description: Chadi's favorite Pokemon! It is said to emerge from darkness to steal the lives of those who become lost in mountains.
Images:
  Photo: images/gengar.jpg
  Typeicon: icons/ghost.jpg
  Weaknessicon: icons/dark.jpg
Moves:
  1. name=Dark Pulse, dp=80, type=dark
  2. name=Double Team, type=normal
  3. name=Shadow Ball, dp=80, type=ghost
  4. name=Venoshock, dp=65, type=poison
Similarity: [0.4219414]
```
