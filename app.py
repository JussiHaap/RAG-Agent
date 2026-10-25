import json
import math
import re
import logging
import time
from functools import wraps

import ollama
from fastapi import FastAPI
from fastapi.responses import FileResponse
from pydantic import BaseModel


# =========================================================
# Configuration
# =========================================================

INDEX_FILE = "recipe_index.json"

EMBEDDING_MODEL = "nomic-embed-text"
LLM_MODEL = "llama3.1:8b"

# Maximum number of retrieval attempts when query
# refinement is enabled.
MAX_SEARCH_ATTEMPTS = 3

# Retrieve and evaluate up to eight candidate sections.
# The UI displays only the top three, plus any supporting
# candidates found below rank three.
RETRIEVAL_RESULTS = 8
DISPLAY_RESULTS = 3

# Console logging for live visibility into retrieval and LLM work.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("recipe_agent")


def timed_chat(stage, **kwargs):
    """Run an Ollama chat call and log its duration."""
    started = time.perf_counter()
    logger.info("LLM START | %s | model=%s", stage, kwargs.get("model", LLM_MODEL))
    try:
        response = ollama.chat(**kwargs)
        elapsed = time.perf_counter() - started
        logger.info("LLM DONE  | %s | %.2f sec", stage, elapsed)
        return response
    except Exception:
        elapsed = time.perf_counter() - started
        logger.exception("LLM ERROR | %s | after %.2f sec", stage, elapsed)
        raise


def log_run_duration(func):
    """Log total duration of an agent run, including early returns/errors."""
    @wraps(func)
    def wrapper(*args, **kwargs):
        started = time.perf_counter()
        logger.info("========== NEW QUESTION ==========")
        try:
            result = func(*args, **kwargs)
            logger.info("AGENT DONE | total=%.2f sec", time.perf_counter() - started)
            return result
        except Exception:
            logger.exception("AGENT ERROR | total=%.2f sec", time.perf_counter() - started)
            raise
        finally:
            logger.info("========== END QUESTION ==========\n")
    return wrapper


# =========================================================
# Load recipe index
# =========================================================

with open(INDEX_FILE, "r", encoding="utf-8") as f:
    documents = json.load(f)


# =========================================================
# FastAPI
# =========================================================

app = FastAPI()


# =========================================================
# Request model
# =========================================================

class QuestionRequest(BaseModel):

    question: str

    evidence_evaluation: bool = False

    query_refinement: bool = False

    source_selection: bool = False


# =========================================================
# Query normalization
# =========================================================

def normalize_query(query):
    """
    Normalize a query before vector retrieval.

    Examples:

        What is good for dessert?
        what is good for dessert
        WHAT IS GOOD FOR DESSERT!

    all become:

        what is good for dessert
    """

    query = query.strip()

    query = query.lower()

    query = re.sub(
        r"[?!.,;:]+$",
        "",
        query
    )

    query = query.strip()

    return query


# =========================================================
# Vector similarity
# =========================================================

def cosine_similarity(a, b):

    dot = sum(
        x * y
        for x, y in zip(a, b)
    )

    magnitude_a = math.sqrt(
        sum(x * x for x in a)
    )

    magnitude_b = math.sqrt(
        sum(x * x for x in b)
    )

    if magnitude_a == 0 or magnitude_b == 0:
        return 0

    return dot / (
        magnitude_a * magnitude_b
    )


# =========================================================
# Retrieve recipe sections
# =========================================================

def retrieve(query):

    normalized_query = normalize_query(
        query
    )

    embed_started = time.perf_counter()
    logger.info("EMBED START | query=%r | model=%s", normalized_query, EMBEDDING_MODEL)
    try:
        response = ollama.embed(
            model=EMBEDDING_MODEL,
            input=normalized_query
        )
        logger.info("EMBED DONE  | %.2f sec", time.perf_counter() - embed_started)
    except Exception:
        logger.exception("EMBED ERROR | after %.2f sec", time.perf_counter() - embed_started)
        raise

    query_embedding = response["embeddings"][0]

    results = []

    for document in documents:

        similarity = cosine_similarity(
            query_embedding,
            document["embedding"]
        )

        results.append({

            "similarity":
                similarity,

            "source":
                document["source"],

            "section":
                document["section"],

            "text":
                document["text"]

        })

    results.sort(
        key=lambda x: x["similarity"],
        reverse=True
    )

    candidate_results = results[:RETRIEVAL_RESULTS]
    logger.info(
        "RETRIEVAL DONE | candidates=%d | top %d: %s",
        len(candidate_results),
        min(DISPLAY_RESULTS, len(candidate_results)),
        "; ".join(
            f"{item['source']} ({item['similarity']:.3f})"
            for item in candidate_results[:DISPLAY_RESULTS]
        ) or "no results"
    )
    return candidate_results


# =========================================================
# Format retrieved information
# =========================================================

def format_context(results):

    context_parts = []

    for i, result in enumerate(results):

        context_parts.append(
            f"""
SOURCE {i + 1}

Recipe/file: {result['source']}
Section: {result['section']}
Similarity: {result['similarity']:.3f}

{result['text']}
"""
        )

    return "\n".join(context_parts)


# =========================================================
# Evaluate retrieved sources
# =========================================================

def evaluate_sources(
    question,
    results
):
    """
    Determine which of the displayed retrieval results actually
    support the user's question.

    The LLM must provide:
      - the source number
      - an exact quote from that source
      - a short explanation of why the quote answers the question

    Python first verifies that the quote exists in the actual source.
    A second LLM verification step then checks whether the quoted text
    genuinely supports the user's question.

    All retrieved candidate sections are eligible, including candidates not displayed in the top three.
    """

    context = format_context(results)

    prompt = f"""
You are an evidence evaluator for a recipe information agent.

Determine which of the RETRIEVED SOURCES actually contain evidence
that can answer the user's specific question.

USER QUESTION:
{question}

RETRIEVED SOURCES:
{context}

STRICT RULES:

1. Evaluate ONLY the retrieved sources shown above. There may be up to eight.

2. Judge the actual source text, not semantic similarity.

3. Do not use general knowledge.

4. Do not invent information.

5. A source supports the question only if its actual text contains
   information that directly answers the user's question.

6. A source being about food, cooking, recipes, or a similar ingredient
   is NOT enough.

7. A source's filename or section title is NOT enough.

8. Do not make common-sense assumptions that are not stated in the
   source text.

9. Every selected source MUST have an exact quote copied from that
   source.

10. The quote must contain the actual evidence relevant to the user's
    question.

11. The explanation must describe the direct connection between the
    quoted text and the user's question.

12. If you cannot provide a direct supporting quote, do not select the
    source.

13. If none of the retrieved sources directly support the question, return
    an empty list.

EXAMPLE 1:
Question: "What are the ingredients for sushi?"
Source text: "400 g salmon fillet, 500 g potatoes, 2 carrots."

This does NOT support the question. Salmon being an ingredient in some
foods does not make this a sushi recipe.

EXAMPLE 2:
Question: "What is good for dessert?"
Source text: "Apple Crumble can be served warm with vanilla ice cream,
custard, or lightly whipped cream."

This DOES support the question because the source explicitly describes
how the recipe can be served as a dessert option.

Return ONLY valid JSON:

{{
  "supporting_sources": [
    {{
      "source": 1,
      "quote": "exact text copied from the source",
      "reason": "brief explanation of why this exact text directly supports the question"
    }}
  ]
}}

If none support the question:

{{
  "supporting_sources": []
}}
"""

    response = timed_chat(
        "evidence evaluation",
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    raw = response["message"]["content"].strip()

    try:
        data = json.loads(raw)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            try:
                data = json.loads(match.group(0))
            except json.JSONDecodeError:
                data = {"supporting_sources": []}
        else:
            data = {"supporting_sources": []}

    source_entries = data.get("supporting_sources", [])

    if not isinstance(source_entries, list):
        return []

    proposed = []

    for entry in source_entries:
        if not isinstance(entry, dict):
            continue

        source_number = entry.get("source")
        quote = entry.get("quote")
        reason = entry.get("reason", "")

        if not isinstance(source_number, int):
            continue

        if not isinstance(quote, str):
            continue

        if not isinstance(reason, str):
            reason = ""

        if not (1 <= source_number <= len(results)):
            continue

        result = results[source_number - 1]

        source_text = re.sub(
            r"\s+",
            " ",
            result["text"]
        ).strip()

        normalized_quote = re.sub(
            r"\s+",
            " ",
            quote
        ).strip()

        if not normalized_quote:
            continue

        # The quote MUST actually exist in the source.
        if normalized_quote.lower() not in source_text.lower():
            print("Rejected supporting source:")
            print(f"  Source: {result['source']}")
            print("  Reason: quoted evidence was not found in source text.")
            continue

        proposed.append({
            "result": result,
            "quote": normalized_quote,
            "reason": reason
        })

    if not proposed:
        return []

    # -----------------------------------------------------
    # Second verification pass
    # -----------------------------------------------------
    # The first LLM proposes evidence. A second, deliberately
    # stricter pass checks whether that evidence actually answers
    # the user's question. This helps prevent cases such as:
    # "sushi" -> "salmon soup contains salmon".
    # -----------------------------------------------------

    verification_parts = []

    for i, item in enumerate(proposed, start=1):
        result = item["result"]
        verification_parts.append(
            f"""
CANDIDATE {i}
Source: {result['source']}
Section: {result['section']}
Exact quoted evidence: {item['quote']}
Evaluator reason: {item['reason']}
Full source text: {result['text']}
"""
        )

    verification_context = "\n".join(verification_parts)

    verification_prompt = f"""
You are the final evidence verifier for a document information agent.

USER QUESTION:
{question}

The first evaluator proposed the following supporting evidence:

{verification_context}

Your job is to reject any source that does NOT directly support the
user's question.

STRICT RULES:

1. Use only the information in the supplied source text.

2. The quoted evidence must directly answer or substantively support
   the user's question.

3. Do NOT use general knowledge.

4. Do NOT infer a connection that is not explicitly supported.

5. Do NOT treat shared ingredients, shared food categories, or vague
   semantic similarity as sufficient evidence.

6. For example, a source containing "salmon" does NOT support a question
   about "sushi" unless the source explicitly connects the salmon to sushi.

7. A source about one recipe does not support a question about another
   recipe merely because the recipes share an ingredient.

8. Be conservative. If there is any meaningful doubt, reject the source.

Return ONLY valid JSON in this exact format:

{{
  "verified_sources": [
    1
  ]
}}

If none genuinely support the question:

{{
  "verified_sources": []
}}
"""

    verification_response = timed_chat(
        "final evidence verification",
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": verification_prompt
            }
        ]
    )

    verification_raw = (
        verification_response["message"]["content"].strip()
    )

    try:
        verification_data = json.loads(verification_raw)
    except json.JSONDecodeError:
        match = re.search(
            r"\{.*\}",
            verification_raw,
            re.DOTALL
        )
        if match:
            try:
                verification_data = json.loads(match.group(0))
            except json.JSONDecodeError:
                verification_data = {"verified_sources": []}
        else:
            verification_data = {"verified_sources": []}

    verified_numbers = verification_data.get(
        "verified_sources",
        []
    )

    if not isinstance(verified_numbers, list):
        return []

    verified = []

    for number in verified_numbers:
        if not isinstance(number, int):
            continue

        if not (1 <= number <= len(proposed)):
            continue

        result = proposed[number - 1]["result"]

        # Avoid duplicates.
        key = (
            result["source"],
            result["section"]
        )

        if any(
            existing["source"] == key[0]
            and existing["section"] == key[1]
            for existing in verified
        ):
            continue

        verified.append(result)

    return verified


# =========================================================
# Query refinement
# =========================================================

def reformulate_query(
    question,
    context
):

    prompt = f"""
You are improving a search query for a recipe document
collection.

Original user question:

{question}


The previous search did not provide enough evidence.

Create ONE concise alternative search query.


Rules:

- Preserve the original user's intent.
- Do not answer the question.
- Do not add new requirements.
- Do not change what the user is asking.
- Use different wording only when it may improve retrieval.
- Keep the query concise.
- Return ONLY the new search query.


Example:

Original:
"What is good for dessert?"

Good:
"dessert recipes"

Bad:
"What desserts are typically served warm?"

The bad example changes the user's intent.


Previous retrieved information:

{context}
"""

    response = timed_chat(
        "query reformulation",
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response["message"]["content"]
        .strip()
    )


# =========================================================
# Generate final answer
# =========================================================

def generate_answer(
    question,
    results
):

    context = format_context(
        results
    )

    prompt = f"""
You are a recipe information assistant.

Answer the user's question using ONLY the information
contained in the provided recipe sources.

Do NOT use general knowledge.

Do NOT invent:

- ingredients
- quantities
- cooking times
- instructions
- serving suggestions
- recipes
- facts


Every factual claim in your answer must be supported by
the supplied source text.

If the user asks for a recommendation, only recommend
recipes supported by the supplied sources.

If the user asks for ingredients, only provide ingredients
that appear in the supplied sources.

If the user asks what is good for dessert, only recommend
recipes that the sources actually support as dessert options.


If the sources genuinely do not contain enough information
to answer the question, respond exactly:

I couldn't find relevant information in the recipe collection.


Otherwise:

- Answer the user's actual question directly.
- Be concise but useful.
- Name the recipe when appropriate.
- Do not mention SOURCE 1, SOURCE 2 or SOURCE 3.
- Do not mention similarity scores.
- Do not discuss the retrieval process.
- Do not invent citations.
- Do not use information outside the supplied sources.


USER QUESTION:

{question}


SUPPORTING RECIPE SOURCES:

{context}
"""

    response = timed_chat(
        "final answer generation",
        model=LLM_MODEL,
        messages=[
            {
                "role": "user",
                "content": prompt
            }
        ]
    )

    return (
        response["message"]["content"]
        .strip()
    )


# =========================================================
# Build source information for UI
# =========================================================

def build_source_list(
    all_results,
    supporting_results,
    show_supporting
):

    if not all_results:
        return []


    supporting_keys = set()


    if show_supporting:

        for result in supporting_results:

            key = (
                result["source"],
                result["section"]
            )

            supporting_keys.add(
                key
            )


    sources = []


    # -----------------------------------------------------
    # Display the same top 3 retrieval results that were
    # eligible for evidence evaluation.
    # -----------------------------------------------------

    for result in all_results[:DISPLAY_RESULTS]:

        key = (
            result["source"],
            result["section"]
        )


        sources.append({

            "source":
                result["source"],

            "section":
                result["section"],

            "score":
                round(
                    result["similarity"],
                    3
                ),

            "supporting":
                (
                    show_supporting
                    and
                    key in supporting_keys
                )

        })


    return sources


# =========================================================
# Build supporting source information
# =========================================================

def build_supporting_sources(
    supporting_results,
    displayed_results
):

    displayed_keys = set()

    for result in displayed_results:

        displayed_keys.add(
            (
                result["source"],
                result["section"]
            )
        )


    additional_sources = []

    for result in supporting_results:

        key = (
            result["source"],
            result["section"]
        )

        # Don't duplicate a source that is already
        # visible in the top 3.
        if key in displayed_keys:
            continue


        additional_sources.append({

            "source":
                result["source"],

            "section":
                result["section"],

            "score":
                round(
                    result["similarity"],
                    3
                ),

            "supporting":
                True

        })


    return additional_sources


# =========================================================
# Run agent
# =========================================================

@log_run_duration
def run_agent(
    question,
    evidence_evaluation,
    query_refinement,
    source_selection
):

    logger.info("QUESTION | %s", question)
    logger.info(
        "OPTIONS | evidence_evaluation=%s, query_refinement=%s, source_selection=%s",
        evidence_evaluation, query_refinement, source_selection
    )
    current_query = question

    attempts = []

    final_results = []

    final_supporting_results = []

    evidence_sufficient = True


    # =====================================================
    # Search / agent loop
    # =====================================================

    for attempt in range(
        1,
        MAX_SEARCH_ATTEMPTS + 1
    ):
        logger.info("SEARCH ATTEMPT %d/%d | query=%r", attempt, MAX_SEARCH_ATTEMPTS, current_query)

        # -------------------------------------------------
        # Retrieve up to eight candidates. The UI later shows
        # only the top three, plus supporting candidates below rank 3.
        # -------------------------------------------------

        results = retrieve(
            current_query
        )

        context = format_context(
            results
        )


        # -------------------------------------------------
        # Evaluate evidence / supporting sources
        # -------------------------------------------------

        needs_source_evaluation = (
            evidence_evaluation
            or
            source_selection
        )


        if needs_source_evaluation:
            logger.info("EVIDENCE CHECK | evaluating retrieved sources")
            supporting_results = evaluate_sources(
                question,
                results
            )

        else:

            supporting_results = []


        # -------------------------------------------------
        # Determine whether evidence is sufficient
        # -------------------------------------------------

        sufficient = (
            len(supporting_results) > 0
        )


        # If evidence evaluation is disabled, retrieval
        # is considered sufficient.
        if not evidence_evaluation:

            sufficient = True


        # -------------------------------------------------
        # Record agent activity
        # -------------------------------------------------

        attempts.append({

            "attempt":
                attempt,

            "query":
                current_query,

            "sufficient":
                sufficient,

            "evaluation_enabled":
                evidence_evaluation

        })


        # -------------------------------------------------
        # Store latest results
        # -------------------------------------------------

        final_results = results

        final_supporting_results = (
            supporting_results
        )


        # -------------------------------------------------
        # Evidence sufficient
        # -------------------------------------------------

        logger.info(
            "EVIDENCE RESULT | sufficient=%s | supporting_sources=%d",
            sufficient, len(supporting_results)
        )

        if sufficient:

            evidence_sufficient = True

            break


        # -------------------------------------------------
        # Evidence insufficient
        # -------------------------------------------------

        evidence_sufficient = False


        # -------------------------------------------------
        # Query refinement disabled
        # -------------------------------------------------

        if not query_refinement:

            break


        # -------------------------------------------------
        # Maximum attempts reached
        # -------------------------------------------------

        if attempt == MAX_SEARCH_ATTEMPTS:

            break


        # -------------------------------------------------
        # Reformulate query
        # -------------------------------------------------

        logger.info("QUERY REFINEMENT | asking LLM for an alternative query")
        current_query = reformulate_query(
            question,
            context
        )
        logger.info("QUERY REFINED | new query=%r", current_query)


    # =====================================================
    # Evidence evaluation says no answer exists
    # =====================================================

    if (
        evidence_evaluation
        and
        not evidence_sufficient
    ):

        displayed_results = (
            final_results[:DISPLAY_RESULTS]
        )

        return {

            "answer":
                "I couldn't find relevant information "
                "in the recipe collection.",

            "sources":
                build_source_list(
                    final_results,
                    final_supporting_results,
                    source_selection
                ),

            "supporting_sources":
                build_supporting_sources(
                    final_supporting_results,
                    displayed_results
                ),

            "attempts":
                attempts

        }


    # =====================================================
    # Source selection enabled
    # =====================================================

    if source_selection:
        logger.info("ANSWER MODE | using only verified supporting sources")

        if not final_supporting_results:

            displayed_results = (
                final_results[:DISPLAY_RESULTS]
            )

            return {

                "answer":
                    "I couldn't find relevant information "
                    "in the recipe collection.",

                "sources":
                    build_source_list(
                        final_results,
                        [],
                        True
                    ),

                "supporting_sources":
                    [],

                "attempts":
                    attempts

            }


        # -------------------------------------------------
        # Generate answer ONLY from sources selected by
        # the evidence evaluator.
        # -------------------------------------------------

        final_answer = generate_answer(
            question,
            final_supporting_results
        )


    # =====================================================
    # Source selection disabled
    # =====================================================

    else:
        logger.info("ANSWER MODE | using top retrieved sources")
        final_answer = generate_answer(
            question,
            final_results
        )


    # =====================================================
    # Display top 3 retrieval results
    # =====================================================

    displayed_results = (
        final_results[:DISPLAY_RESULTS]
    )


    # =====================================================
    # Build UI source lists
    # =====================================================

    visible_sources = build_source_list(
        final_results,
        final_supporting_results,
        source_selection
    )


    additional_supporting_sources = (
        build_supporting_sources(
            final_supporting_results,
            displayed_results
        )
    )


    # =====================================================
    # Return
    # =====================================================

    return {

        "answer":
            final_answer,

        "sources":
            visible_sources,

        "supporting_sources":
            additional_supporting_sources,

        "attempts":
            attempts

    }


# =========================================================
# Routes
# =========================================================

@app.get("/")
def home():

    return FileResponse(
        "static/index.html"
    )


@app.post("/ask")
def ask(
    request: QuestionRequest
):

    return run_agent(

        question=
            request.question,

        evidence_evaluation=
            request.evidence_evaluation,

        query_refinement=
            request.query_refinement,

        source_selection=
            request.source_selection

    )