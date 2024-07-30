from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from openai import OpenAI

# Step 1: Load the trained model and tokenizer
model_path = "./trained_model"
api_key = "2fJZxqtyEVZ3xhzwfmY1ZfDDS8zecGV1"
base_url = "https://api.deepinfra.com/v1/openai"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

# If the tokenizer does not have a padding token defined, set it
if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained(model_path)

# Step 2: Define a function to generate text
def generate_text(prompt, model, tokenizer, max_length=230, num_return_sequences=1):
    # Encode the input
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()

    # Generate text using advanced sampling techniques
    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,        # Avoid repeating n-grams
        early_stopping=True,           # Stop early if it seems appropriate
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,               # Control creativity
        top_k=50,                      # Limit sampling to the top 50 tokens
        top_p=0.95,                    # Nucleus sampling
        repetition_penalty=1.2         # Penalize excessive repetition
    )

    # Decode and return the generated outputs
    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]

# Step 3: Define a function to use the Deepinfra API to improve the slogan
def improve_eslogan(eslogan, api_key, base_url, num_options=4):
    openai = OpenAI(api_key=api_key, base_url=base_url)
    prompt = f"Eslogan: {eslogan}\nGenera {num_options} opciones de eslogan mejorados."
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    improved_eslogans = chat_completion.choices[0].message.content.strip().split('\n')[:num_options]
    return improved_eslogans

# Step 4: Test the model with an input prompt
prompt = "Company: EcoClean Segment: Families Emotion: Elegance Description: Super detergent"
generated_texts = generate_text(prompt, model, tokenizer)

# API key and base URL for Deepinfra


# Improved regular expression to capture variations of "Slogan"
eslogan_regex = r"Eslogan[:,| : ](.*)"

# Display only the information after "Slogan:"
for i, text in enumerate(generated_texts):
    match = re.search(eslogan_regex, text)
    if match:
        eslogan = match.group(1).strip()
        improved_eslogans = improve_eslogan(eslogan, api_key, base_url)
        print(f"Eslógan Original {i+1}:\n{eslogan}\n")
        print(f"Opciones de Eslógan Mejorado {i+1}:")
        for idx, option in enumerate(improved_eslogans):
            print(f"Opción: {idx+1}: {option}")
        print()
    else:
        print(f"El texto generado {i+1} no contiene la palabra 'Eslogan'.\n{text}\n")
