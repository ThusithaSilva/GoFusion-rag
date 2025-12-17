from transformers import AutoTokenizer, AutoModelForCausalLM

def main():
    model_name = "distilgpt2"  # small & safe for CPU

    print("Loading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    print("Loading model...")
    model = AutoModelForCausalLM.from_pretrained(model_name)

    prompt = "are you happy"

    print("Generating output...")
    inputs = tokenizer(prompt, return_tensors="pt")
    outputs = model.generate(
        **inputs,
        max_new_tokens=80,
        do_sample=True,
        temperature=0.8,
        top_p=0.95,
        pad_token_id=tokenizer.eos_token_id,
    )

    result = tokenizer.decode(outputs[0], skip_special_tokens=True)
    print("\nModel Output:\n")
    print(result)

if __name__ == "__main__":
    main()
