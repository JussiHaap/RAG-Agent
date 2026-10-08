import ollama

text = "A simple recipe for tomato soup."

response = ollama.embed(
    model="nomic-embed-text",
    input=text
)

embedding = response["embeddings"][0]

print("Embedding created successfully.")
print("Number of dimensions:", len(embedding))
print("First 5 values:", embedding[:5])