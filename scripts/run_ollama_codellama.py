import requests
from src.rag.loader import load_document
from src.rag.chunker import chunk_text
from src.rag.retriever import retrieve
from src.rag.prompts import build_prompt


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:latest"

question = "What did Thusitha eat this morning?"

# Load knowledge base
doc = load_document("data/documents/personal_facts.txt")
chunks = chunk_text(doc)


# WITHOUT RAG
prompt_no_rag = build_prompt(question)

# WITH RAG
context = retrieve(chunks)
prompt_with_rag = build_prompt(question, context)


def generate(prompt):
    payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "stream": False
    }

    response = requests.post(OLLAMA_URL, json=payload)
    response.raise_for_status()
    return response.json()["response"]

def main():
    # prompt = "Write a simple Go function that adds two integers."

    output = generate(prompt_with_rag)
    print("\nModel Output:\n")
    print(output)

if __name__ == "__main__":
    main()
