import json
import torch
from torch.utils.data import Dataset
import wandb
from transformers import AutoTokenizer, AutoModelForCausalLM, DataCollatorForSeq2Seq, Seq2SeqTrainingArguments, Seq2SeqTrainer, TrainerCallback

class PrintProgressCallback(TrainerCallback):
    def __init__(self, print_interval=100):
        super().__init__()
        self.print_interval = print_interval

    def on_step_end(self, args, state, control, **kwargs):
        if state.global_step % self.print_interval == 0:
            progress = 100 * (state.global_step / state.max_steps)
            print(f"Step {state.global_step}/{state.max_steps} ({progress:.2f}%)")

class Text2TextGenerationDataset(Dataset):
    def __init__(self, data, tokenizer):
        self.data = data
        self.tokenizer = tokenizer

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]
        input_text = example["input_text"]
        target_text = example["target_text"]

        inputs = self.tokenizer(input_text, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors="pt")
        targets = self.tokenizer(target_text, truncation=True, padding='max_length', max_length=self.tokenizer.model_max_length, return_tensors="pt")

        # Squeeze the input tensors to remove the extra dimension
        inputs = {key: value.squeeze(0) for key, value in inputs.items()}
        targets = {key: value.squeeze(0) for key, value in targets.items()}

        # Add labels to the inputs
        inputs["labels"] = targets["input_ids"].clone()

        return {**inputs, **targets}


class WandbCustomCallback(TrainerCallback):
    def on_log(self, args, state, control, model=None, logs=None, **kwargs):
        if state.is_local_process_zero and logs is not None:
            wandb.log({k: v for k, v in logs.items() if not k.startswith("train_")})

print("CUDA available:", torch.cuda.is_available())
print("Number of GPUs:", torch.cuda.device_count())
print("Current device:", torch.cuda.current_device())
print("GPU name:", torch.cuda.get_device_name(torch.cuda.current_device()))

wandb.login()
run = wandb.init(project="fine_tuning_bloom_de", name="1st_resl_run")

#Load training data from the JSON file
with open("alpaca_data_de.json", "r") as f:
    train_data = json.load(f)

print(f"Loaded {len(train_data)} training examples")

#Load the bigscience/bloom tokenizer and model
tokenizer = AutoTokenizer.from_pretrained("bigscience/bloom-1b1")
model = AutoModelForCausalLM.from_pretrained("bigscience/bloom-1b1")
print("LOADED... bigscience/bloom-1b1")

#Preprocess and tokenize the data
def preprocess(example):
    input_text = example["instruction"] + ": " + example["input"]
    return {"input_text": input_text, "target_text": example["output"]}

tokenizer.model_max_length = 512 # Set max length for tokenizer
tokenized_train_data = [preprocess(example) for example in train_data]
train_dataset = Text2TextGenerationDataset(tokenized_train_data, tokenizer)

#Set up the data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

#Set training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir="fine_tuned_bloom_de",
    overwrite_output_dir=True,
    num_train_epochs=3,
    per_device_train_batch_size=4,
    learning_rate=2e-5,
    weight_decay=0,
    save_strategy="epoch",
    save_total_limit=2,
    fp16=True,
    report_to="wandb", # Report training metrics to wandb
)

#Create the trainer and fine-tune the model
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    data_collator=data_collator,
    tokenizer=tokenizer,
    callbacks=[WandbCustomCallback(), PrintProgressCallback(print_interval=100)], # Add the PrintProgressCallback
)

print("Starting training...")
trainer.train()
print("Training complete")

#Close the wandb run
wandb.finish()