import json
import math
import ollama

INDEX_FILE = "recipe_index.json"
EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

MAX_SEARCH_ATTEMPTS = 2
TOP_RESULTS = 3


def cosine_similarity(a, b):
    dot = sum(x * y for x, y in zip(a, b))
    magnitude_a = math.sqrt(sum(x * x for x in a))
    magnitude_b = math.sqrt(sum(x * x for x in b))

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot / (magnitude_a * magnitude_b)


def retrieve(query, documents):
    """Search the recipe collection and return the most relevant sections."""

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=query
    )

    query_embedding = response["embeddings"][0]

    results = []

    for document in documents:
        similarity = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({
            "similarity": similarity,
            "source": document["source"],
            "section": document["section"],
            "text": document["text"]
        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    return results[:TOP_RESULTS]


def format_context(results):
    """Convert retrieved sections into context for the LLM."""

    context_parts = []

    for i, result in enumerate(results):
        context_parts.append(
            f"""
SOURCE {i + 1}
File: {result['source']}
Section: {result['section']}

{result['text']}
"""
        )

    return "\n".join(context_parts)


def evaluate_evidence(question, context):
    """Ask the LLM whether the retrieved evidence is sufficient."""

    prompt = f"""
You are evaluating retrieved recipe information.

USER QUESTION:
{question}

RETRIEVED INFORMATION:
{context}

Determine whether the retrieved information contains enough
relevant information to answer the user's question accurately.

Respond with ONLY one of these two words:

YES

or

NO
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    decision = response["message"]["content"].strip().upper()

    if "YES" in decision:
        return True

    return False


def reformulate_query(question, context):
    """Ask the LLM to create a better search query while preserving intent."""

    prompt = f"""
You are improving a search query for a recipe document collection.

The user's original question is:

{question}

The previous search did not provide enough evidence.

Create ONE concise alternative search query.

IMPORTANT RULES:
- Preserve the exact intent of the user's original question.
- Do not add new requirements or assumptions.
- Do not change the type of thing the user is asking for.
- Use alternative or more specific wording only when it may improve retrieval.
- The query should help find relevant recipes in the collection.
- Do not answer the question.

Return ONLY the new search query.

Example:

Original question:
What is good for dessert?

Good reformulation:
dessert recipes

Bad reformulation:
What desserts are typically served warm?

Previous retrieved information:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"].strip()


def generate_answer(question, results):
    """Generate the final answer using retrieved evidence."""

    context = format_context(results)

    prompt = f"""
You are a recipe information assistant.

Answer the user's question using ONLY the information contained
in the provided recipe sources.

Do not invent ingredients, quantities, instructions, or other facts.

If the sources do not contain enough information to answer the
question, clearly say that the information was not found in the
recipe collection.

Give a concise and useful natural-language answer.

At the end, provide the source file and section used for the answer.

USER QUESTION:
{question}

RECIPE SOURCES:
{context}
"""

    response = ollama.chat(
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return response["message"]["content"]


# ---------------------------------------------------------
# Main program
# ---------------------------------------------------------

with open(INDEX_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


question = input("\nAsk a question: ")

current_query = question
final_results = None


for attempt in range(1, MAX_SEARCH_ATTEMPTS + 1):

    print(f"\n--- Agent search attempt {attempt} ---")
    print(f"Search query: {current_query}")

    # Retrieve information
    results = retrieve(
        current_query,
        documents
    )

    # Show retrieved sources
    print("\nRetrieved sources:\n")

    for i, result in enumerate(results):
        print(f"[{i + 1}] {result['source']}")
        print(f"    Section: {result['section']}")
        print(f"    Similarity: {result['similarity']:.3f}")
        print()

    # Create context
    context = format_context(results)

    # Agent evaluates the evidence
    print("Agent is evaluating the retrieved evidence...")

    sufficient = evaluate_evidence(
        question,
        context
    )

    if sufficient:
        print("Agent decision: SUFFICIENT")
        final_results = results
        break

    print("Agent decision: INSUFFICIENT")

    # If this was the final attempt, stop
    if attempt == MAX_SEARCH_ATTEMPTS:
        final_results = results
        break

    # Agent decides what search to perform next
    print("Agent is reformulating the search query...")

    current_query = reformulate_query(
        question,
        context
    )

    print(f"New search query: {current_query}")


# Generate final answer
print("\n--- Final answer ---\n")

answer = generate_answer(
    question,
    final_results
)

print(answer)