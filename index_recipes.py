from pathlib import Path
import json
import re
import ollama


RECIPES_DIR = Path("recipes")
OUTPUT_FILE = "recipe_index.json"

EMBEDDING_MODEL = "nomic-embed-text"


def split_into_sections(text):
    """
    Split a recipe into sections based on headings.

    Supports both:

    ## Ingredients

    and:

    Ingredients
    ===========
    """

    lines = text.splitlines()

    sections = []

    current_title = "Introduction"
    current_lines = []

    for line in lines:

        stripped = line.strip()

        # Markdown heading
        if re.match(r"^#{1,3}\s+", stripped):

            if current_lines:
                sections.append({
                    "section": current_title,
                    "text": "\n".join(current_lines).strip()
                })

            current_title = re.sub(
                r"^#{1,3}\s+",
                "",
                stripped
            )

            current_lines = []

        # Plain-text heading
        elif (
            stripped
            and current_lines
            and re.match(r"^[=\-]{3,}$", stripped)
        ):

            # This is an underline for the previous line,
            # so the previous line becomes the section title.

            current_title = current_lines[-1].strip()

            current_lines = current_lines[:-1]

        else:
            current_lines.append(line)

    # Add final section
    if current_lines:
        sections.append({
            "section": current_title,
            "text": "\n".join(current_lines).strip()
        })

    return [
        section
        for section in sections
        if section["text"]
    ]


def create_embedding(text):

    response = ollama.embed(
        model=EMBEDDING_MODEL,
        input=text
    )

    return response["embeddings"][0]


documents = []

print("Reading recipe files...\n")


for file_path in sorted(RECIPES_DIR.iterdir()):

    if file_path.suffix.lower() not in [".txt", ".md"]:
        continue

    print(f"Processing: {file_path.name}")

    text = file_path.read_text(
        encoding="utf-8"
    )

    sections = split_into_sections(text)

    print(f"  Found {len(sections)} sections")

    for section_number, section in enumerate(sections):

        # Include the section title in the embedded text.
        # This helps retrieval understand the context.
        searchable_text = (
            f"Recipe: {file_path.stem}\n"
            f"Section: {section['section']}\n\n"
            f"{section['text']}"
        )

        print(
            f"  Embedding: {section['section']}"
        )

        embedding = create_embedding(
            searchable_text
        )

        documents.append({
            "source": file_path.name,
            "section": section["section"],
            "section_number": section_number,
            "text": section["text"],
            "embedding": embedding
        })


with open(
    OUTPUT_FILE,
    "w",
    encoding="utf-8"
) as f:

    json.dump(
        documents,
        f,
        ensure_ascii=False
    )


print("\nIndexing complete.")
print(f"Sections indexed: {len(documents)}")
print(f"Saved to: {OUTPUT_FILE}")