import json
from hyperdb import HyperDB

# Load documents from the JSONL file
documents = []

with open("demo/pokemon.jsonl", "r") as f:
    for line in f:
        documents.append(json.loads(line))

# Instantiate HyperDB with the list of documents and the key "description"
db = HyperDB(documents, key="info.description")

# Save the HyperDB instance to a file
db.save("demo/pokemon_hyperdb.pickle.gz")

# Load the HyperDB instance from the file
db.load("demo/pokemon_hyperdb.pickle.gz")

# Query the HyperDB instance with a text input
results = db.query("Likes to sleep.", top_k=5)

# Define a function to pretty print the results
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

# Print the top 5 most similar Pokémon descriptions
def print_pokemon_info(results):
    for res in results:
        if len(res) == 2:
            document, similarity = res
        elif len(res) == 1:
            document = res[0]
            similarity = None
        else:
            print("Invalid result format.")
            continue
        print(format_entry(document, similarity))  # Pretty-print the Pokémon data
        print("-" * 40)  # Add a separator between entries

# Display the query results
print_pokemon_info(results)
