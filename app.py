from flask import Flask, render_template, request
from transformers import GPT2LMHeadModel, GPT2Tokenizer
import re
from openai import OpenAI

app = Flask(__name__)

# Load the trained GPT-2 model and tokenizer
model_path = "./trained_model"
tokenizer = GPT2Tokenizer.from_pretrained(model_path)

if tokenizer.pad_token_id is None:
    tokenizer.pad_token_id = tokenizer.eos_token_id

model = GPT2LMHeadModel.from_pretrained(model_path)


# Define a function to generate text
def generate_text(prompt, model, tokenizer, max_length=230, num_return_sequences=1):
    inputs = tokenizer.encode(prompt, return_tensors='pt')
    attention_mask = inputs.ne(tokenizer.pad_token_id).long()

    outputs = model.generate(
        inputs,
        attention_mask=attention_mask,
        max_length=max_length,
        num_return_sequences=num_return_sequences,
        no_repeat_ngram_size=2,
        early_stopping=True,
        pad_token_id=tokenizer.eos_token_id,
        temperature=0.7,
        top_k=50,
        top_p=0.95,
        repetition_penalty=1.2
    )

    return [tokenizer.decode(output, skip_special_tokens=True) for output in outputs]


# Define a function to use the Deepinfra API to improve the slogan
# Define a function to use the Deepinfra API to improve the slogan
def improve_eslogan(prompt, eslogan, api_key, base_url, num_options=4):
    openai = OpenAI(api_key=api_key, base_url=base_url)
    prompt = f"Con esta info: {prompt} y este eslogan: {eslogan} \nGenera {num_options} opciones para el eslogan."
    chat_completion = openai.chat.completions.create(
        model="meta-llama/Meta-Llama-3.1-8B-Instruct",
        messages=[{"role": "user", "content": prompt}],
    )
    response_text = chat_completion.choices[0].message.content.strip()

    # Use regex to capture only the eslogans
    improved_eslogans = []
    matches = re.findall(r"\d+\.\s*\*\*(.*?)\*\*", response_text)

    for match in matches[:num_options]:
        improved_eslogans.append(match.strip())

    return improved_eslogans

# Flask routes
@app.route('/')
def index():
    return render_template('index.html')


@app.route('/generate', methods=['POST'])
def generate():
    company_name = request.form['company_name']
    segment = request.form['segment']
    emotion = request.form['emotion']
    description = request.form['description']

    prompt = f"Company: {company_name} Segment: {segment} Emotion: {emotion} Description: {description}"
    generated_texts = "xd"#generate_text(prompt, model, tokenizer)

    # API key and base URL for Deepinfra
    api_key = "2fJZxqtyEVZ3xhzwfmY1ZfDDS8zecGV1"
    base_url = "https://api.deepinfra.com/v1/openai"

    # Improved regular expression to capture variations of "Slogan"
    eslogan_regex = r"Eslogan[:,| : ](.*)"

    results = []
    for i, text in enumerate(generated_texts):
        match = re.search(eslogan_regex, text)
        if match:

            #eslogan = match.group(1).strip()
            #improved_eslogans = improve_eslogan(prompt,eslogan, api_key, base_url)
            eslogan = "xd"
            improved_eslogans = "xd"
            results.append({
                'original': eslogan,
                'improved': improved_eslogans
            })
        else:
            results.append({
                'original': None,
                'improved': ["El texto generado no contiene 'Eslogan' en la salida, revise la información ingresada."]
            })

    return render_template('index.html', results=results, prompt=prompt)


if __name__ == '__main__':
    app.run(debug=True)