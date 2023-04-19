import json
import time
import torch
from transformers import MarianMTModel, MarianTokenizer
from concurrent.futures import ThreadPoolExecutor

model_name = "Helsinki-NLP/opus-mt-en-de"
model = MarianMTModel.from_pretrained(model_name)
tokenizer = MarianTokenizer.from_pretrained(model_name)

# Use CPU
device = torch.device("cpu")
model.to(device)

# Load the JSON file
with open("alpaca_data.json", "r") as f:
    data = json.load(f)

def translate_obj(obj):
    instruction_text = obj["instruction"]
    input_text = obj["input"]
    output_text = obj["output"]

    # Translate the instruction, input, and output
    translated_instruction = translate_text(instruction_text)
    translated_input = translate_text(input_text)
    translated_output = translate_text(output_text)

    return {
        "instruction": translated_instruction,
        "input": translated_input,
        "output": translated_output
    }

def translate_text(input_text):
    input_text = "".join(filter(lambda x: x.isprintable(), input_text))  # Remove non-printable characters
    if input_text.strip() == "":
        # Return an empty string if the input text is empty
        return ""
    else:
        # Tokenize the input text
        input_tokens = tokenizer(input_text, return_tensors="pt", truncation=True, max_length=512).to(device)

        # Generate the translated text
        translated = model.generate(input_ids=input_tokens["input_ids"], attention_mask=input_tokens["attention_mask"], max_new_tokens=512)

        # Decode the translated text
        output_text = tokenizer.decode(translated[0], skip_special_tokens=True)

        return output_text

# Set the maximum number of workers
max_workers = 4

# Process the translations in parallel
translated_data = []
start_time = time.time()
with ThreadPoolExecutor(max_workers=max_workers) as executor:
    futures = []
    for obj in data:
        futures.append(executor.submit(translate_obj, obj))

    # Collect the results as they become ready
    for i, future in enumerate(futures):
        translated_data.append(future.result())

        # Print progress update
        progress = (i+1) / len(data) * 100
        elapsed_time = time.time() - start_time
        time_per_obj = elapsed_time / (i+1)
        print(f"Translated {i+1} of {len(data)} objects ({progress:.2f}% complete). Average time per object: {time_per_obj:.2f}s")

        # Save the translated data periodically
        if (i+1) % 100 == 0:
            with open("translated_alpaca_data.json", "w") as f:
                json.dump(translated_data, f, indent=4)

# Save the final translated data to a JSON file with the same formatting as the input file
with open("translated_alpaca_data.json", "w") as f:
    json.dump(translated_data, f, indent=4, sort_keys=True)
