import json
import torch
from datasets import Dataset
from transformers import (
    AutoModelForCausalLM,
    AutoTokenizer,
    TrainingArguments,
    Trainer,
    DataCollatorForLanguageModeling
)
from peft import LoraConfig, get_peft_model, TaskType

# Configuration
MODEL_NAME = "/projects/project1/laceytt/companion_bot/models/opt-1.3b"
DATA_DIR = "/projects/project1/laceytt/companion_bot/data"
OUTPUT_DIR = "/projects/project1/laceytt/companion_bot/checkpoints/opt-1.3b-empathetic-lora-exp2"

## Configuration of Training Hyperparameters
MAX_LENGTH = 512      # max length of input
BATCH_SIZE = 4        
EPOCHS = 2
LEARNING_RATE = 2e-4
WARMUP_STEPS = 100

# Configuration of LoRA Hyperparameters
R = 16
LORA_ALPHA = 32
LORA_DROPOUT = 0.1

# load preprocessed data
def load_json_data(filepath):
    with open(filepath, 'r') as f:
        return json.load(f)

train_data = load_json_data(f"{DATA_DIR}/emphaphetic_dialogues_finetune_train.json")
valid_data = load_json_data(f"{DATA_DIR}/emphaphetic_dialogues_finetune_valid.json")

train_dataset = Dataset.from_list(train_data)
valid_dataset = Dataset.from_list(valid_data)

print(f"Train size: {len(train_dataset)}")
print(f"Valid size: {len(valid_dataset)}")

# load model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
tokenizer.pad_token = tokenizer.eos_token # use end-of-sequence as pad token

model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    torch_dtype=torch.float16,
    device_map="auto"
)

# configure LoRA
lora_config = LoraConfig(
    task_type=TaskType.CAUSAL_LM,
    r=R,
    lora_alpha=LORA_ALPHA,
    lora_dropout=LORA_DROPOUT,
    target_modules=["q_proj", "v_proj"]  # attention layers of OPT
)

# Transform OPT model into a model that can be fine-tuned with LoRA and print trainable parameters
model = get_peft_model(model, lora_config)
model.print_trainable_parameters()

# data pre-processing
def tokenize_function(examples):
    return tokenizer(
        examples["text"],
        truncation=True,
        max_length=MAX_LENGTH,
        padding="max_length"
    )

train_dataset = train_dataset.map(tokenize_function, batched=True, remove_columns=["text"])
valid_dataset = valid_dataset.map(tokenize_function, batched=True, remove_columns=["text"])

# configuration training arguments
training_args = TrainingArguments(
    output_dir=OUTPUT_DIR,
    num_train_epochs=EPOCHS,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    warmup_steps=WARMUP_STEPS,
    learning_rate=LEARNING_RATE,
    fp16=True,
    logging_steps=50, # for printing loss every 50 steps
    eval_strategy="steps",
    eval_steps=200,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    load_best_model_at_end=True,
    report_to="none"
)

data_collator = DataCollatorForLanguageModeling(tokenizer=tokenizer, mlm=False)

# train
trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=valid_dataset,
    data_collator=data_collator
)

print("=" * 50)
print("Hyperparameters:")
print(f"  EPOCHS: {EPOCHS}")
print(f"  BATCH_SIZE: {BATCH_SIZE}")
print(f"  LEARNING_RATE: {LEARNING_RATE}")
print(f"  MAX_LENGTH: {MAX_LENGTH}")
print(f"  WARMUP_STEPS: {WARMUP_STEPS}")
print(f"  LoRA r: {R}")
print(f"  LoRA alpha: {LORA_ALPHA}")
print(f"  LoRA dropout: {LORA_DROPOUT}")
print("=" * 50)
print("Starting training...")
trainer.train()

# save model
model.save_pretrained(OUTPUT_DIR)
tokenizer.save_pretrained(OUTPUT_DIR)
print(f"Model saved to {OUTPUT_DIR}")