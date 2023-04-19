import openai
import json
import time

# Load your OpenAI API key
openai.api_key = "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"

# Read the user_oriented_instructions.jsonl file
instructions = []
with open("/Users/nicolaiklutke/Desktop/ChatGPTSeminar/03_Testing/user_oriented_instructions_de.jsonl", "r") as f:
    for line in f:
        data = json.loads(line)
        instructions.append(data)

# Function to send a prompt to InstructGPT Ada and get the response
def get_instructgpt_response(prompt):
    response = openai.Completion.create(
        engine="text-babbage-001", #text-ada-001
        prompt=prompt,
        max_tokens=100,
        n=1,
        stop=None,
        temperature=0.5,
    )

    return response.choices[0].text.strip()

# Save the responses to instructgpt-ada_predictions.jsonl
def save_responses(responses):
    with open("/Users/nicolaiklutke/Desktop/ChatGPTSeminar/03_Testing/instructgpt-babbage_predictions_de.jsonl", "w") as f:
        for response in responses:
            f.write(json.dumps(response) + "\n")

# Iterate through the instructions and get the InstructGPT Ada response for each input
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

    # Add a delay to avoid hitting the rate limit
    time.sleep(1)  # Sleep for 1 second

# Save final results
save_responses(responses)
print("All results saved.")
