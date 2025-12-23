import matplotlib.pyplot as plt
import numpy as np

# Data
models = [
    "RoBERTa-base", "DistilBERT", "DeBERTa-v3", "BERT-base", "ALBERT-base",
    "LSTM (RNN)", "SVM", "Naive Bayes", "Random Forest", "KNN"
]

accuracy = [98.68, 98.47, 98.17, 97.99, 97.58, 87.00, 85.93, 84.01, 83.57, 58.62]
macro_precision = [98.68, 98.47, 98.17, 97.99, 97.61, 87.00, 86.00, 84.00, 84.00, 69.00]
macro_recall = [98.70, 98.47, 98.17, 98.01, 97.58, 87.00, 86.00, 84.00, 84.00, 59.00]

# --- Figure 4.4: Accuracy Comparison ---
plt.figure(figsize=(10, 6))
bars = plt.bar(models, accuracy, color='skyblue', edgecolor='black')

# Add values on top of bars
for bar in bars:
    yval = bar.get_height()
    plt.text(bar.get_x() + bar.get_width()/2, yval + 1, f'{yval}%', ha='center', va='bottom', fontsize=9)

plt.xlabel('Models', fontsize=12)
plt.ylabel('Accuracy (%)', fontsize=12)
plt.xticks(rotation=45, ha='right')
plt.ylim(0, 110) # Extend y-axis to fit labels
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.tight_layout()
plt.savefig('Figure_4.4_Accuracy_Comparison.png')
plt.show()

# --- Figure 4.5: Macro-Precision and Macro-Recall Comparison ---
x = np.arange(len(models))
width = 0.35

plt.figure(figsize=(12, 6))
rects1 = plt.bar(x - width/2, macro_precision, width, label='Macro-Precision', color='lightcoral', edgecolor='black')
rects2 = plt.bar(x + width/2, macro_recall, width, label='Macro-Recall', color='lightgreen', edgecolor='black')

# Add labels, title and custom x-axis tick labels
plt.xlabel('Models', fontsize=12)
plt.ylabel('Percentage (%)', fontsize=12)
plt.xticks(x, models, rotation=45, ha='right')
plt.legend()
plt.ylim(0, 115) # Extend y-axis to fit labels
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Function to add labels on top of bars
def autolabel(rects):
    for rect in rects:
        height = rect.get_height()
        plt.annotate(f'{height}%',
                    xy=(rect.get_x() + rect.get_width() / 2, height),
                    xytext=(0, 3),  # 3 points vertical offset
                    textcoords="offset points",
                    ha='center', va='bottom', rotation=90, fontsize=8)

autolabel(rects1)
autolabel(rects2)

plt.tight_layout()
plt.savefig('Figure_4.5_Macro_Metrics_Comparison.png')
plt.show()