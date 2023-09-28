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
def format_entry(pokemon, score):
    name = pokemon["name"]
    hp = pokemon["hp"]
    info = pokemon["info"]
    pokedex_id = info["id"]
    pkm_type = info["type"]
    weakness = info["weakness"]
    description = info["description"]

    prettify_pokemon = f"""Name: {name}
Pokedex ID: {pokedex_id}
HP: {hp}
Type: {pkm_type}
Weakness: {weakness}
Description: {description}
"""
    
    if score is not None:
        prettify_pokemon += f"Similarity score: {score}\n"
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
