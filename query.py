from retriever import Retriever
from context import build_context
from prompt import build_rag_prompt
from generator import generate_answer


# Create retriever
retriever = Retriever()


# User question
query = input(
    "\nEnter your question: "
)


# -----------------------------
# Retrieval
# -----------------------------

results = retriever.search(
    query,
    top_k=3
)


# -----------------------------
# Context construction
# -----------------------------

context = build_context(
    results
)


# -----------------------------
# RAG prompt
# -----------------------------

prompt = build_rag_prompt(
    query,
    context
)


# -----------------------------
# LLM generation
# -----------------------------

answer = generate_answer(
    prompt
)


# -----------------------------
# Display
# -----------------------------

print("\n" + "=" * 70)
print("QUESTION")
print("=" * 70)

print(query)


print("\n" + "=" * 70)
print("ANSWER")
print("=" * 70)

print(answer)