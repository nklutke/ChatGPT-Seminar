import json
import time
from transformers import AutoTokenizer, AutoModelForCausalLM

def generate_response(model, tokenizer, prompt):
    input_ids = tokenizer.encode(prompt, return_tensors="pt")
    output = model.generate(input_ids, max_length=200, num_return_sequences=1)
    decoded_output = tokenizer.decode(output[0][input_ids.shape[-1]:], skip_special_tokens=True)
    return decoded_output

tokenizer_file = "ftbloom"
config_file = "ftbloom/config.json"
model_path = "ftbloom"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(tokenizer_file, use_fast=True, config_file=config_file)
model = AutoModelForCausalLM.from_pretrained(model_path, config=config_file)

# Read the user_oriented_instructions.jsonl file
instructions = []
with open("user_oriented_instructions_de.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        instructions.append(data)

# Function to send a prompt to your fine-tuned model and get the response
def get_instructgpt_response(prompt):
    return generate_response(model, tokenizer, prompt)

# Save the responses to fine_tuned_bloom_de_predictions.jsonl
def save_responses(responses):
    with open("fine_tuned_bloom_predictions_de.jsonl", "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")

# Iterate through the instructions and get your fine-tuned model's response for each input
responses = []
total_instructions = len(instructions)
save_interval = 10

for index, instruction_data in enumerate(instructions):
    input_text = instruction_data["instances"][0]["input"]
    if input_text == "":
        input_section = "" 
    else:
        input_section = f"\n\nEingabe: {input_text}"
    prompt = f"{instruction_data['instruction']}{input_section}\nAusgabe:"
    response = get_instructgpt_response(prompt)
    result = {
        "prompt": prompt,
        "instruction": instruction_data["instruction"],
        "input": input_text,
        "response": response,
        "target": instruction_data["instances"][0]["output"],
    }
    responses.append(result)

    # Print progress
    progress = (index + 1) / total_instructions * 100
    print(f"Progress: {progress:.2f}%")

    # Save intermediate results
    if (index + 1) % save_interval == 0:
        save_responses(responses)
        print("Intermediate results saved.")

# Save final results
save_responses(responses)
print("All results saved.")
