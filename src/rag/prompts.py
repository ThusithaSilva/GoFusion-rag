def build_prompt(question, context=None):
    if context:
        context_block = "\n".join(context)
        return f"""
You are a factual assistant.
Answer ONLY using the provided context.
If the answer is not in the context, say "I don't know".

Context:
{context_block}

Question:
{question}

Answer:
"""
    else:
        return f"""
You are a factual assistant.
If you do not know the answer, say "I don't know".

Question:
{question}

Answer:
"""
