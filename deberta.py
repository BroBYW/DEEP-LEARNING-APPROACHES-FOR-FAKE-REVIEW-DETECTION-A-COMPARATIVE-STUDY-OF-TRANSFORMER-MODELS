import pandas as pd
from sklearn.model_selection import train_test_split
from transformers import AutoTokenizer, AutoModelForSequenceClassification, Trainer, TrainingArguments, DebertaV2Tokenizer
from datasets import Dataset
from transformers import EarlyStoppingCallback
import json
import torch

LEARNING_RATE = 2e-5
BATCH_SIZE = 4  # Reduced batch size to fit within memory
PATIENCE = 4
EPOCH = 999999999999

# Load the dataset
data_path_train = "train_data.csv"
data_path_val = "val_data.csv"
train_df = pd.read_csv(data_path_train, sep=",", names=["category", "rating", "label", "text"], encoding="ISO-8859-1")
val_df = pd.read_csv(data_path_val, sep=",", names=["category", "rating", "label", "text"], encoding="ISO-8859-1")

train_df = train_df.dropna()
val_df = val_df.dropna()

# Generate label-to-ID mapping dynamically
unique_labels = pd.concat([train_df["label"], val_df["label"]]).unique()
label_mapping = {label: idx for idx, label in enumerate(unique_labels)}

# Save label-to-ID mapping to a file
label_mapping_file = "label_mapping.json"
with open(label_mapping_file, "w") as f:
    json.dump(label_mapping, f)

print(f"Label-to-ID mapping saved to {label_mapping_file}")

# Convert labels to integers for model compatibility
train_df["label"] = train_df["label"].map(label_mapping)
val_df["label"] = val_df["label"].map(label_mapping)

# Prepare the dataset for the model
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer and model (prefer safetensors-capable repo)
model_name = "microsoft/deberta-v3-base"
# Force slow tokenizer explicitly (requires 'sentencepiece').
tokenizer = DebertaV2Tokenizer.from_pretrained(model_name)
model = AutoModelForSequenceClassification.from_pretrained(
    model_name,
    num_labels=len(label_mapping),
    use_safetensors=True,
    ignore_mismatched_sizes=True,
)

# Tokenize the datasets
def tokenize_function(examples):
    return tokenizer(examples["text"], padding="max_length", truncation=True)

train_dataset = train_dataset.map(tokenize_function, batched=True)
val_dataset = val_dataset.map(tokenize_function, batched=True)

# Remove unnecessary columns
train_dataset = train_dataset.remove_columns(["text", "category", "rating"])
val_dataset = val_dataset.remove_columns(["text", "category", "rating"])

train_dataset.set_format("torch")
val_dataset.set_format("torch")

# Define training arguments with gradient accumulation, mixed precision, and smaller batch size
training_args = TrainingArguments(
    output_dir="./results",
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,  # Reduced batch size to fit GPU memory
    per_device_eval_batch_size=BATCH_SIZE,  # Reduced batch size for evaluation
    num_train_epochs=EPOCH,
    weight_decay=0.01,
    logging_dir="./logs",  # Directory for logging
    logging_steps=10,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="accuracy",
    remove_unused_columns=False,
    fp16=True,  # Enable mixed precision training to save memory
    gradient_accumulation_steps=4,  # Accumulate gradients over 4 steps
    disable_tqdm=True
)

# Free up GPU memory before starting the training
torch.cuda.empty_cache()

# Define the trainer
from sklearn.metrics import accuracy_score, precision_recall_fscore_support

def compute_metrics(eval_pred):
    logits, labels = eval_pred
    predictions = logits.argmax(axis=-1)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, predictions, average="weighted", zero_division=1)
    acc = accuracy_score(labels, predictions)
    return {"accuracy": acc, "precision": precision, "recall": recall, "f1": f1}

trainer = Trainer(
    model=model,
    args=training_args,
    train_dataset=train_dataset,
    eval_dataset=val_dataset,
    tokenizer=tokenizer,
    compute_metrics=compute_metrics,
    callbacks=[EarlyStoppingCallback(early_stopping_patience=PATIENCE)]
)

# Train the model
trainer.train()

# Save the fine-tuned model
model.save_pretrained("./debertatrained_model")
tokenizer.save_pretrained("./debertatrained_model")
