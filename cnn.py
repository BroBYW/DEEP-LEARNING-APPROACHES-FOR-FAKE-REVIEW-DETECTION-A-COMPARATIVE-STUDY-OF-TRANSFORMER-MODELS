import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten, Conv1D, MaxPooling1D, Embedding
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

# Load your custom CSV dataset
df = pd.read_csv('fake reviews dataset.csv')  # Replace with your file path

# Convert 'label' column to numeric labels using Label Encoding
label_encoder = LabelEncoder()
df['label'] = label_encoder.fit_transform(df['label'])  # CG -> 0, OR -> 1 (or more categories)

# Ensure the labels are integers
labels = df['label'].values

# Extract the 'text' column
texts = df['text'].astype(str).tolist()  # Ensure all texts are strings

# Tokenizing the text data
top_words = 7000
tokenizer = Tokenizer(num_words=top_words)
tokenizer.fit_on_texts(texts)
sequences = tokenizer.texts_to_sequences(texts)

# Padding the sequences to ensure consistent input size
max_words = 450
X = pad_sequences(sequences, maxlen=max_words)

# Split the dataset into 80% training and 20% testing
# Split the dataset into 70% training, 15% validation, and 15% testing
X_temp, X_test, y_temp, y_test = train_test_split(X, labels, test_size=0.15, random_state=42)
X_train, X_val, y_train, y_val = train_test_split(X_temp, y_temp, test_size=0.176, random_state=42)  # 0.176 of 85% is roughly 15%

# Build the CNN model
model = Sequential()

# Embedding layer to convert words to dense vectors
model.add(Embedding(input_dim=top_words, output_dim=32, input_length=max_words))

# Convolutional layer to capture n-gram patterns in the text
model.add(Conv1D(32, 3, padding='same', activation='relu'))

# MaxPooling layer to down-sample the output
model.add(MaxPooling1D())

# Flatten the output to feed into the fully connected layers
model.add(Flatten())

# Fully connected layer for the classification task
model.add(Dense(250, activation='relu'))

# Output layer: Sigmoid for binary classification (use softmax for multi-class)
model.add(Dense(1, activation='sigmoid'))  # Change to 'softmax' for multi-class

# Compile the model with binary crossentropy loss for binary classification
model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])

# Print model summary
model.summary()

# Import callbacks for early stopping
from tensorflow.keras.callbacks import EarlyStopping

# Create early stopping callback
early_stopping = EarlyStopping(
    monitor='val_loss',
    patience=4,
    verbose=1,
    restore_best_weights=True
)

# Train the model
history = model.fit(
    X_train, 
    y_train, 
    validation_data=(X_val, y_val),  # Use validation set instead of test set
    epochs=10, 
    batch_size=128, 
    verbose=2,
    callbacks=[early_stopping]
)

# Evaluate the model on the test data
y_pred = model.predict(X_test)
y_pred = (y_pred > 0.5)  # Convert probabilities to binary values (0 or 1)

# Print classification report
print("\nClassification Report:")
print(classification_report(y_test, y_pred))

# Print confusion matrix
print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)

# Plot confusion matrix
sns.heatmap(cm, annot=True, fmt='g', cmap='Blues', xticklabels=label_encoder.classes_, yticklabels=label_encoder.classes_)
plt.xlabel('Predicted')
plt.ylabel('True')
plt.title('Confusion Matrix')
plt.show()

# Calculate metrics from confusion matrix
TN = cm[0][0]  # True Negatives
FP = cm[0][1]  # False Positives
FN = cm[1][0]  # False Negatives
TP = cm[1][1]  # True Positives

# Calculate Accuracy
accuracy = (TP + TN) / (TP + TN + FP + FN)
print(f"\nAccuracy (from confusion matrix): {accuracy:.4f}")

# Calculate Precision
precision = TP / (TP + FP) if (TP + FP) != 0 else 0
print(f"Precision (from confusion matrix): {precision:.4f}")

# Calculate Recall
recall = TP / (TP + FN) if (TP + FN) != 0 else 0
print(f"Recall (from confusion matrix): {recall:.4f}")

# Calculate F1-Score
f1_score = 2 * (precision * recall) / (precision + recall) if (precision + recall) != 0 else 0
print(f"F1-Score (from confusion matrix): {f1_score:.4f}")

# Calculate True Negative Rate (TNR)
TNR = TN / (TN + FP) if (TN + FP) != 0 else 0
print(f"True Negative Rate (TNR): {TNR:.4f}")
