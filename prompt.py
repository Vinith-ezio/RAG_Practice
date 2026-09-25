def build_rag_prompt(question, context):

    prompt = f"""
You are a question-answering assistant.

Use the provided context to answer the user's question.

Rules:
1. Answer only using the provided context.
2. Do not invent information.
3. If the context does not contain the answer, say:
   "I don't have enough information in the provided context."
4. Give a concise and clear answer.

CONTEXT:
{context}

QUESTION:
{question}

ANSWER:
"""

    return prompt