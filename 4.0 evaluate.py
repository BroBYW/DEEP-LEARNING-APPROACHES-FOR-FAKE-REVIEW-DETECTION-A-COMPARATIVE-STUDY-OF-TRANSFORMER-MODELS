import pandas as pd
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
from sklearn.metrics import precision_score, recall_score, accuracy_score, confusion_matrix
import seaborn as sns
import matplotlib.pyplot as plt
import json
import numpy as np

# Paths to model and test data
test_data_path = "test_data.csv"
model_path = "./debertatrained_model"

# Load the fine-tuned model and tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)

# Load the label mapping
label_mapping_file = "label_mapping.json"
with open(label_mapping_file, "r") as f:
    label_mapping = json.load(f)

# Reverse the label mapping to map IDs back to labels
id_to_label = {v: k for k, v in label_mapping.items()}
# Filter out only the actual class names (CG and OR) for display
class_names = [name for name in label_mapping.keys() if name != "label"]
num_classes = len(class_names)

# Load the test dataset
df = pd.read_csv(test_data_path, sep=",", names=["label", "text"], encoding="ISO-8859-1")
df = df.dropna()

# Convert labels to their corresponding IDs using the loaded mapping
df["label"] = df["label"].map(label_mapping)

# Ensure no NaN values after mapping
df = df.dropna()

# Function to make predictions in batches for efficiency
def predict_batch(texts, batch_size=16):
    predictions = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i:i + batch_size]
        inputs = tokenizer(batch_texts, return_tensors="pt", padding=True, truncation=True, max_length=512).to(device)
        
        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            predicted_classes = torch.argmax(logits, dim=-1).cpu().numpy()
            predictions.extend(predicted_classes)
    
    return predictions

# Perform evaluation
texts = df["text"].tolist()
true_labels = df["label"].tolist()

# Predict in batches
predicted_labels = predict_batch(texts)

# Convert lists to NumPy arrays for correct boolean operations
true_labels = np.array(true_labels)
predicted_labels = np.array(predicted_labels)

# Calculate accuracy
accuracy = accuracy_score(true_labels, predicted_labels)

# Calculate precision for each class (handling zero division)
precision = precision_score(true_labels, predicted_labels, average=None, zero_division=0)

# Calculate recall for each class
recall = recall_score(true_labels, predicted_labels, average=None, zero_division=0)

# Calculate F1-score for each class
f1_scores = 2 * (precision * recall) / (precision + recall)
# Handle division by zero cases
f1_scores = np.nan_to_num(f1_scores, 0)  # Replace NaN with 0

# Calculate true negative rate (specificity) for each class
# Filter the confusion matrix to only include the classes we want (CG and OR)
class_indices = [label_mapping[name] for name in class_names]
# Create a filtered confusion matrix with only the classes we want
filtered_conf_matrix = np.zeros((len(class_indices), len(class_indices)))
for i, true_idx in enumerate(class_indices):
    for j, pred_idx in enumerate(class_indices):
        # Count occurrences where true label is true_idx and predicted label is pred_idx
        filtered_conf_matrix[i, j] = np.sum((true_labels == true_idx) & (predicted_labels == pred_idx))

conf_matrix = filtered_conf_matrix.astype(int)  # Convert to integers for display
true_negatives = []
false_positives = []
for i in range(len(class_indices)):
    true_negative = np.sum(conf_matrix[np.arange(len(class_indices)) != i, np.arange(len(class_indices)) != i])
    false_positive = np.sum(conf_matrix[np.arange(len(class_indices)) != i, i])
    true_negatives.append(true_negative)
    false_positives.append(false_positive)

true_negative_rate = [tn / (tn + fp) if (tn + fp) > 0 else 0 for tn, fp in zip(true_negatives, false_positives)]

# Display results
print(f"Accuracy: {accuracy:.4f}")
print(f"Precision per class: {dict(zip(class_names, precision))}")
print(f"Recall per class: {dict(zip(class_names, recall))}")
print(f"F1-score per class: {dict(zip(class_names, f1_scores))}")
print(f"True Negative Rate per class: {dict(zip(class_names, true_negative_rate))}")

# Display the confusion matrix using seaborn
plt.figure(figsize=(len(class_indices), len(class_indices)))
sns.heatmap(conf_matrix, annot=True, fmt='d', cmap="Blues", xticklabels=class_names, yticklabels=class_names)
plt.xlabel("True")
plt.ylabel("Predicted")
plt.title("Confusion Matrix")
plt.show()