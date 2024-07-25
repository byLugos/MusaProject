# Step 1: Import necessary libraries
from transformers import GPT2Tokenizer, GPT2LMHeadModel, Trainer, TrainingArguments, TextDataset, DataCollatorForLanguageModeling

# Step 2: Configure the GPT-2 tokenizer and model
model_name = "gpt2-medium"
tokenizer = GPT2Tokenizer.from_pretrained(model_name)
model = GPT2LMHeadModel.from_pretrained(model_name)

# Step 3: Prepare the dataset for training
train_dataset = TextDataset(
    tokenizer=tokenizer,
    file_path="EntrenoNum4.txt",  # Path to the labeled .txt file
    block_size=128
)

data_collator = DataCollatorForLanguageModeling(
    tokenizer=tokenizer,
    mlm=False,  # Masked Language Modeling is disabled
)

# Step 4: Configure training arguments
training_args = TrainingArguments(
    output_dir="./results",                    # Directory for saving results
    overwrite_output_dir=True,                 # Overwrite the content of the output directory
    num_train_epochs=3,                        # Number of training epochs
    per_device_train_batch_size=8,             # Batch size for training
    save_steps=5000,                           # Save the model every 5000 steps
    save_total_limit=3,                        # Limit the total number of saved checkpoints to 3
    fp16=True,                                 # Use mixed precision training if possible
    weight_decay=0.01,                         # Apply weight decay to prevent overfitting
    logging_steps=500,                         # Log every 500 steps
    logging_dir="./logs",                      # Directory for saving logs
    evaluation_strategy="steps",               # Evaluate every certain number of steps
    eval_steps=5000,                           # Evaluate the model every 5000 steps
    save_strategy="steps",                     # Strategy for saving checkpoints
    load_best_model_at_end=True,               # Load the best model at the end of training
    metric_for_best_model="eval_loss",         # Metric to determine the best model
    greater_is_better=False                    # Indicates that a lower loss is better
)

# Step 5: Train the model
trainer = Trainer(
    model=model,
    args=training_args,
    data_collator=data_collator,
    train_dataset=train_dataset,
)

trainer.train()

# Save the trained model
trainer.save_model("./trained_model")
tokenizer.save_pretrained("./trained_model")
