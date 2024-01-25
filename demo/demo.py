import json
from hyperdb import HyperDB

# Load documents from the JSONL file
documents = []

with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Instantiate HyperDB with the list of documents and the key "description"
db = HyperDB(documents)

# Save the HyperDB instance to a file
db.save("demo/pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the file
db.load("demo/pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance with a text input
results = db.query("Likes to sleep.", top_k=5)

# Define a function to pretty print the results
def format_entry(pokemon, score=None, index=None):
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
    # Print the original index of the document in the database (not the ranking index)
    if index is not None:
        prettify_pokemon += f"\nIndex: {index}"

    return prettify_pokemon

# Function to print query results
def print_pokemon_info(results, show_similarity=True, show_index=True):
    for res in results:
        # Check if the result contains similarity and index information
        if isinstance(res, tuple) and len(res) == 3:
            document, similarity, index = res
        else:  # When return_similarity is False, 'res' is directly the document
            document = res
            similarity = None
            index = None

        # Decide whether to display similarity and/or index based on function arguments
        similarity_to_display = similarity if show_similarity and similarity is not None else None
        index_to_display = index if show_index and index is not None else None

        print(format_entry(document, similarity_to_display, index_to_display))  # Pretty-print the Pokémon data
        print("-" * 40)  # Add a separator between entries

# Display the query results with customized settings, show_similarity and show_index only have an effect when return_similarity=True in the query method
print_pokemon_info(results, show_similarity=True, show_index=False)
