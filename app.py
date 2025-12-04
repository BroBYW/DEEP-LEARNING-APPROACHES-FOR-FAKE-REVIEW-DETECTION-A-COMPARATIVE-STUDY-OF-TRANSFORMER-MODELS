from flask import Flask, render_template, request, jsonify, send_file
from transformers import AutoTokenizer, AutoModelForSequenceClassification
import torch
import json
import pandas as pd
import io
import csv
from datetime import datetime
import os
import random

app = Flask(__name__)

# Load the fine-tuned model and tokenizer
model_path = "./robertatrained_model"
tokenizer = AutoTokenizer.from_pretrained(model_path)
model = AutoModelForSequenceClassification.from_pretrained(model_path)

# Load the label mapping
label_mapping_file = "label_mapping.json"
with open(label_mapping_file, "r") as f:
    label_mapping = json.load(f)

# Reverse the label mapping to map IDs back to labels
id_to_label = {v: k for k, v in label_mapping.items()}

# Define user-friendly labels
label_descriptions = {
    "label": "Genuine",
    "CG": "Computer Generated (Fake)",
    "OR": "Original Review (Genuine)"
}

def predict_text(text):
    """Make prediction on a single text"""
    if not text.strip():
        return None, 0.0, {}
    
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
    
    # Get all probabilities for detailed results
    all_probabilities = {
        id_to_label[i]: probabilities[0, i].item() 
        for i in range(len(id_to_label))
    }
    
    return predicted_label, predicted_probability, all_probabilities

@app.route('/')
def index():
    """Main page"""
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    """API endpoint for single text prediction"""
    try:
        data = request.get_json()
        text = data.get('text', '').strip()
        
        if not text:
            return jsonify({'error': 'No text provided'}), 400
        
        predicted_label, confidence, all_probs = predict_text(text)
        
        if predicted_label is None:
            return jsonify({'error': 'Unable to process text'}), 400
        
        # Determine if it's fake or genuine
        is_fake = predicted_label == 'CG'  # Only CG is fake
        result_type = 'fake' if is_fake else 'genuine'
        
        # Only show the two main categories: CG (fake) and OR (genuine)
        # Combine "label" probability with "OR" since both are genuine
        main_probabilities = {}
        cg_prob = all_probs.get('CG', 0.0)
        or_prob = all_probs.get('OR', 0.0)
        label_prob = all_probs.get('label', 0.0)
        
        # Combine OR and label probabilities as they're both genuine
        genuine_prob = or_prob + label_prob
        
        main_probabilities['Computer Generated (Fake)'] = round(cg_prob * 100, 2)
        main_probabilities['Original Review (Genuine)'] = round(genuine_prob * 100, 2)
        
        response = {
            'predicted_label': predicted_label,
            'description': label_descriptions[predicted_label],
            'confidence': round(confidence * 100, 2),
            'result_type': result_type,
            'all_probabilities': main_probabilities
        }
        
        return jsonify(response)
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/batch_upload', methods=['POST'])
def batch_upload():
    """Handle batch file upload"""
    try:
        if 'file' not in request.files:
            return jsonify({'error': 'No file uploaded'}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({'error': 'No file selected'}), 400
        
        # Read the file
        if file.filename.endswith('.csv'):
            df = pd.read_csv(file)
            
            # Look for text columns with various possible names
            possible_text_columns = ['text_', 'text', 'review', 'review_text', 'content', 'message', 'comment']
            text_column = None
            
            for col in possible_text_columns:
                if col in df.columns:
                    text_column = col
                    break
            
            if text_column is None:
                available_columns = ', '.join(df.columns.tolist())
                return jsonify({
                    'error': f'CSV must have a text column. Available columns: {available_columns}. Expected one of: {", ".join(possible_text_columns)}'
                }), 400
                
            texts = df[text_column].tolist()
        elif file.filename.endswith('.txt'):
            content = file.read().decode('utf-8')
            texts = [line.strip() for line in content.split('\n') if line.strip()]
        else:
            return jsonify({'error': 'Only CSV and TXT files are supported'}), 400
        
        # Process each text
        results = []
        for i, text in enumerate(texts):
            predicted_label, confidence, all_probs = predict_text(text)
            if predicted_label is not None:
                is_fake = predicted_label == 'CG'  # Only CG is fake
                results.append({
                    'index': i + 1,
                    'text': text[:100] + '...' if len(text) > 100 else text,
                    'full_text': text,
                    'predicted_label': predicted_label,
                    'description': label_descriptions[predicted_label],
                    'confidence': round(confidence * 100, 2),
                    'result_type': 'fake' if is_fake else 'genuine'
                })
        
        return jsonify({'results': results, 'total_processed': len(results)})
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/download_results', methods=['POST'])
def download_results():
    """Download batch results as CSV"""
    try:
        data = request.get_json()
        results = data.get('results', [])
        
        if not results:
            return jsonify({'error': 'No results to download'}), 400
        
        # Create CSV content
        output = io.StringIO()
        writer = csv.writer(output)
        
        # Write header
        writer.writerow(['Index', 'Text', 'Predicted_Label', 'Description', 'Confidence_%', 'Result_Type'])
        
        # Write data
        for result in results:
            writer.writerow([
                result['index'],
                result['full_text'],
                result['predicted_label'],
                result['description'],
                result['confidence'],
                result['result_type']
            ])
        
        # Create file-like object
        output.seek(0)
        
        # Generate filename with timestamp
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"fake_review_detection_results_{timestamp}.csv"
        
        # Create a BytesIO object for the response
        mem = io.BytesIO()
        mem.write(output.getvalue().encode('utf-8'))
        mem.seek(0)
        
        return send_file(
            mem,
            as_attachment=True,
            download_name=filename,
            mimetype='text/csv'
        )
    
    except Exception as e:
        return jsonify({'error': str(e)}), 500

@app.route('/get_fake_example')
def get_fake_example():
    """Get a random fake (CG) review from test_data.csv"""
    try:
        # Read the test data
        df = pd.read_csv('test_data.csv')
        # Detect the text column name robustly
        possible_text_columns = ['text_', 'text', 'review', 'review_text', 'content', 'message', 'comment']
        text_column = None
        for col in possible_text_columns:
            if col in df.columns:
                text_column = col
                break
        if text_column is None:
            available_columns = ', '.join(df.columns.tolist())
            return jsonify({'error': f"CSV must have a text column. Available columns: {available_columns}. Expected one of: {', '.join(possible_text_columns)}"}), 400
        
        # Filter for CG (Computer Generated) reviews
        fake_reviews = df[df['label'] == 'CG']
        
        if len(fake_reviews) == 0:
            return jsonify({'error': 'No fake examples found'}), 404
        
        # Randomly select one review
        random_review = fake_reviews.sample(n=1).iloc[0]
        
        return jsonify({
            'text': random_review[text_column],
            'category': random_review['category'],
            'rating': random_review['rating']
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to load fake example: {str(e)}'}), 500

@app.route('/get_genuine_example')
def get_genuine_example():
    """Get a random genuine (OR) review from test_data.csv"""
    try:
        # Read the test data
        df = pd.read_csv('test_data.csv')
        # Detect the text column name robustly
        possible_text_columns = ['text_', 'text', 'review', 'review_text', 'content', 'message', 'comment']
        text_column = None
        for col in possible_text_columns:
            if col in df.columns:
                text_column = col
                break
        if text_column is None:
            available_columns = ', '.join(df.columns.tolist())
            return jsonify({'error': f"CSV must have a text column. Available columns: {available_columns}. Expected one of: {', '.join(possible_text_columns)}"}), 400
        
        # Filter for OR (Original Review) reviews
        genuine_reviews = df[df['label'] == 'OR']
        
        if len(genuine_reviews) == 0:
            return jsonify({'error': 'No genuine examples found'}), 404
        
        # Randomly select one review
        random_review = genuine_reviews.sample(n=1).iloc[0]
        
        return jsonify({
            'text': random_review[text_column],
            'category': random_review['category'],
            'rating': random_review['rating']
        })
    
    except Exception as e:
        return jsonify({'error': f'Failed to load genuine example: {str(e)}'}), 500

if __name__ == '__main__':
    app.run(debug=True, host='0.0.0.0', port=5000)