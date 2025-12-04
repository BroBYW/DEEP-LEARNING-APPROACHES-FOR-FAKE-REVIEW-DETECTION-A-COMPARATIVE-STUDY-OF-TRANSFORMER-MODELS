from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json

# Load the fine-tuned model and tokenizer
model_path= "./robertatrained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the label mapping
label_mapping_file = "label_mapping.json"
with open(label_mapping_file, "r") as f:
    label_mapping = json.load(f)

# Reverse the label mapping to map IDs back to labels
id_to_label = {v: k for k, v in label_mapping.items()}

# Function to make predictions
def predict_text(text):
    # Tokenize the input text
    inputs = tokenizer(text, return_tensors="pt", padding="max_length", truncation=True, max_length=512)
    
    # Perform inference
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits
        probabilities = torch.nn.functional.softmax(logits, dim=-1)
        predicted_class = torch.argmax(probabilities, dim=-1).item()
    
    # Map predicted class back to label
    predicted_label = id_to_label[predicted_class]
    
    # Extract the probability of the predicted class
    predicted_probability = probabilities[0, predicted_class].item()
    
    return predicted_label, predicted_probability

# Example usage
text = "This vacuum cleaner is the best purchase I have made. It is lightweight and easy to use. The battery lasts a long time, and my floors have never been cleaner. It is also very comfortable to wear and keeps my hands warm in the winter. I highly recommend this product to anyone looking for a great addition to their home. My cat loves this shampoo as well, and it has improved her fur quality. I will definitely buy this again!"
predicted_label, predicted_probability = predict_text(text)

print(f"Predicted Label: {predicted_label}")
print(f"Predicted Probability: {predicted_probability:.4f}")