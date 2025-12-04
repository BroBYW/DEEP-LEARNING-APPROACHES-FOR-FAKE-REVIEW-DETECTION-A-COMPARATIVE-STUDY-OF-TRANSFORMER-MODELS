import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sklearn.preprocessing import LabelEncoder
from collections import defaultdict
from nltk.corpus import wordnet as wn
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn import model_selection, naive_bayes, svm
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Set Random seed
np.random.seed(500)

# Add the Data using pandas
Corpus = pd.read_csv("fake reviews dataset.csv", encoding='latin-1')

# Step - 1: Data Pre-processing - This will help in getting better results through the classification algorithms

# Step - 1a : Remove blank rows if any.
Corpus['text'].dropna(inplace=True)

# Step - 1b : Change all the text to lower case. This is required as python interprets 'dog' and 'DOG' differently
Corpus['text'] = [entry.lower() for entry in Corpus['text']]

# Step - 1c : Tokenization : In this each entry in the corpus will be broken into a set of words
Corpus['text'] = [word_tokenize(entry) for entry in Corpus['text']]

# Step - 1d : Remove Stop words, Non-Numeric and perform Word Stemming/Lemmatization.
# WordNetLemmatizer requires Pos tags to understand if the word is noun, verb, adjective, etc. By default, it is set to Noun
tag_map = defaultdict(lambda: wn.NOUN)
tag_map['J'] = wn.ADJ
tag_map['V'] = wn.VERB
tag_map['R'] = wn.ADV

for index, entry in enumerate(Corpus['text']):
    # Declaring Empty List to store the words that follow the rules for this step
    Final_words = []
    # Initializing WordNetLemmatizer()
    word_Lemmatized = WordNetLemmatizer()
    # pos_tag function below will provide the 'tag' i.e if the word is Noun(N), Verb(V), or something else.
    for word, tag in pos_tag(entry):
        # Below condition is to check for Stop words and consider only alphabets
        if word not in stopwords.words('english') and word.isalpha():
            word_Final = word_Lemmatized.lemmatize(word, tag_map[tag[0]])
            Final_words.append(word_Final)
    # The final processed set of words for each iteration will be stored in 'text_final'
    Corpus.loc[index, 'text_final'] = str(Final_words)

# Step - 2: Split the model into Train, Validation and Test Data sets
# First split: 70% train, 30% remaining
Train_X, temp_X, Train_Y, temp_Y = model_selection.train_test_split(Corpus['text_final'], Corpus['label'], test_size=0.3, random_state=500)

# Second split: Split the remaining 30% into two equal parts (15% each for validation and test)
Val_X, Test_X, Val_Y, Test_Y = model_selection.train_test_split(temp_X, temp_Y, test_size=0.5, random_state=500)

# Step - 3: Label encode the target variable
Encoder = LabelEncoder()
Train_Y = Encoder.fit_transform(Train_Y)
Val_Y = Encoder.transform(Val_Y)  # Use transform instead of fit_transform for validation
Test_Y = Encoder.transform(Test_Y)  # Use transform instead of fit_transform for test

# Step - 4: Vectorize the words by using TF-IDF Vectorizer
Tfidf_vect = TfidfVectorizer(max_features=5000)
Tfidf_vect.fit(Corpus['text_final'])

Train_X_Tfidf = Tfidf_vect.transform(Train_X)
Val_X_Tfidf = Tfidf_vect.transform(Val_X)
Test_X_Tfidf = Tfidf_vect.transform(Test_X)

# Step - 5: Train and Evaluate Models

# Classifier - Naive Bayes
Naive = naive_bayes.MultinomialNB()
Naive.fit(Train_X_Tfidf, Train_Y)

# First evaluate on validation set
val_predictions_NB = Naive.predict(Val_X_Tfidf)
print("Naive Bayes Validation Accuracy Score -> ", accuracy_score(val_predictions_NB, Val_Y) * 100)
print("Naive Bayes Validation Classification Report:\n", classification_report(Val_Y, val_predictions_NB))

# Then evaluate on test set
test_predictions_NB = Naive.predict(Test_X_Tfidf)
print("Naive Bayes Test Accuracy Score -> ", accuracy_score(test_predictions_NB, Test_Y) * 100)
print("Naive Bayes Test Classification Report:\n", classification_report(Test_Y, test_predictions_NB))

# Confusion Matrix for Naive Bayes
cm_NB = confusion_matrix(Test_Y, test_predictions_NB)
TN_NB, FP_NB, FN_NB, TP_NB = cm_NB.ravel()
specificity_NB = TN_NB / (TN_NB + FP_NB)
print(f"Naive Bayes True Negative Rate (Specificity): {specificity_NB}")

# Classifier - SVM
SVM = svm.SVC(C=1.0, kernel='linear', degree=3, gamma='auto')
SVM.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
predictions_SVM = SVM.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("SVM Accuracy Score -> ", accuracy_score(predictions_SVM, Test_Y) * 100)
print("SVM Classification Report:\n", classification_report(Test_Y, predictions_SVM))

# Confusion Matrix for SVM
cm_SVM = confusion_matrix(Test_Y, predictions_SVM)
TN_SVM, FP_SVM, FN_SVM, TP_SVM = cm_SVM.ravel()
specificity_SVM = TN_SVM / (TN_SVM + FP_SVM)
print(f"SVM True Negative Rate (Specificity): {specificity_SVM}")

# Classifier - KNN (K-Nearest Neighbors)
knn = KNeighborsClassifier(n_neighbors=5)
knn.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
predictions_knn = knn.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("KNN Accuracy Score -> ", accuracy_score(predictions_knn, Test_Y) * 100)
print("KNN Classification Report:\n", classification_report(Test_Y, predictions_knn))

# Confusion Matrix for KNN
cm_knn = confusion_matrix(Test_Y, predictions_knn)
TN_knn, FP_knn, FN_knn, TP_knn = cm_knn.ravel()
specificity_knn = TN_knn / (TN_knn + FP_knn)
print(f"KNN True Negative Rate (Specificity): {specificity_knn}")

# Classifier - Random Forest
rf = RandomForestClassifier(n_estimators=100, random_state=42)
rf.fit(Train_X_Tfidf, Train_Y)

# predict the labels on validation dataset
predictions_rf = rf.predict(Test_X_Tfidf)

# Use accuracy_score function to get the accuracy
print("Random Forest Accuracy Score -> ", accuracy_score(predictions_rf, Test_Y) * 100)
print("Random Forest Classification Report:\n", classification_report(Test_Y, predictions_rf))

# Confusion Matrix for Random Forest
cm_rf = confusion_matrix(Test_Y, predictions_rf)
TN_rf, FP_rf, FN_rf, TP_rf = cm_rf.ravel()
specificity_rf = TN_rf / (TN_rf + FP_rf)
print(f"Random Forest True Negative Rate (Specificity): {specificity_rf}")

# Visualization Functions

# Confusion Matrix Visualization function
def plot_confusion_matrix(cm, model_name):
    plt.figure(figsize=(6,6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', cbar=False, 
                xticklabels=['Negative', 'Positive'], yticklabels=['Negative', 'Positive'])
    plt.title(f'Confusion Matrix - {model_name}')
    plt.ylabel('Actual')
    plt.xlabel('Predicted')
    plt.show()

# Plot Confusion Matrices for each model
plot_confusion_matrix(cm_NB, 'Naive Bayes')
plot_confusion_matrix(cm_SVM, 'SVM')
plot_confusion_matrix(cm_knn, 'KNN')
plot_confusion_matrix(cm_rf, 'Random Forest')

# Accuracy Comparison Plot
models = ['Naive Bayes', 'SVM', 'KNN', 'Random Forest']
accuracies = [accuracy_score(predictions_NB, Test_Y) * 100, 
              accuracy_score(predictions_SVM, Test_Y) * 100,
              accuracy_score(predictions_knn, Test_Y) * 100,
              accuracy_score(predictions_rf, Test_Y) * 100]

plt.figure(figsize=(10,6))
plt.bar(models, accuracies, color=['blue', 'green', 'orange', 'red'])
plt.title('Model Accuracy Comparison')
plt.ylabel('Accuracy (%)')
plt.xlabel('Model')
plt.show()

# Classification Report Heatmaps for each model
def classification_report_heatmap(model_name, y_true, y_pred):
    report = classification_report(y_true, y_pred, output_dict=True)
    report_df = pd.DataFrame(report).transpose()
    plt.figure(figsize=(8, 6))
    sns.heatmap(report_df.iloc[:-1, :-1], annot=True, cmap='coolwarm', fmt='.2f')
    plt.title(f'Classification Report Heatmap - {model_name}')
    plt.show()

# Classification Report Heatmaps for each model
classification_report_heatmap('Naive Bayes', Test_Y, predictions_NB)
classification_report_heatmap('SVM', Test_Y, predictions_SVM)
classification_report_heatmap('KNN', Test_Y, predictions_knn)
classification_report_heatmap('Random Forest', Test_Y, predictions_rf)
