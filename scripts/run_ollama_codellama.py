import requests


OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "codellama:latest"

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
    prompt = "Write a simple Go function that adds two integers."

    output = generate(prompt)
    print("\nModel Output:\n")
    print(output)

if __name__ == "__main__":
    main()
