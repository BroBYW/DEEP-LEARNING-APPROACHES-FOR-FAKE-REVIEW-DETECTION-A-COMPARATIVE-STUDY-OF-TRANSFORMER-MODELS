# Fake Review Detection (Text Classification)

This project trains and serves text classifiers to detect computer-generated (fake) reviews versus original (genuine) reviews. It includes training scripts, evaluation utilities, and a Flask web app with single prediction and batch upload.

## Overview
- Two main classes: `CG` (Computer Generated/Fake) and `OR` (Original/Genuine).
- Flask app at `app.py` provides endpoints for single prediction, batch upload, and example sampling.
- Training uses Hugging Face Transformers, safe weight loading via `safetensors`, and early stopping.

## Project Structure
- `1.0 split_data.py` – splits dataset into train/val/test CSVs.
- `2.0 train.py` – fine-tunes a transformer model and saves checkpoints in `results/`.
- `3.0 inference.py` – CLI/script inference helper.
- `4.0 evaluate.py` – computes metrics on validation/test sets.
- `app.py` – Flask web application.
- `trained_model/` – final exported model (config, tokenizer, `model.safetensors`).
- `results/` – intermediate training checkpoints (`checkpoint-<step>/`).
- `label_mapping.json` – mapping between class labels and IDs.
- `templates/index.html` – web UI.

## Installation
1. Create and activate a Python virtual environment.
2. Install dependencies:
   - `pip install -r requirements.txt`
   - If you use the DeBERTa v3 slow tokenizer, also install `sentencepiece`: `pip install sentencepiece`

## Running the Web App
- Start the server: `python app.py`
- Open `http://localhost:5000/`
- Features:
  - Single text prediction with confidence.
  - Batch CSV/TXT upload; auto-detects text column among `['text_', 'text', 'review', 'review_text', 'content', 'message', 'comment']`.
  - Buttons to fetch a random fake (`/get_fake_example`) or genuine (`/get_genuine_example`) sample from `test_data.csv`.

## Training & Checkpoints
- Configure model and training args in `2.0 train.py`.
- Checkpoints are saved under `results/` (e.g., `results/checkpoint-5307/`).
- Resume training: in your script call `trainer.train(resume_from_checkpoint=True)` or specify a path.
- Early stopping: patience is set to `4` by default; you can adjust based on validation stability.

## Model Results
The following metrics are taken from `Model Notes.txt` in this repository.

### Transformer Models
- **BERT-base-uncased** (`./trained_model`)
  - Accuracy: `0.9799`
  - Train loss: `0.0502`, Epochs: `6`
  - TNR per class: `{'label': 1.0, 'CG': 0.9736, 'OR': 0.9863}`

- **RoBERTa-base** (`./robertatrained_model`)
  - Accuracy: `0.9868`
  - Train loss: `0.0453`, Epochs: `16`
  - TNR per class: `{'label': 1.0, 'CG': 0.9870, 'OR': 0.9867}`

- **DistilBERT-base-uncased** (`./distillberttrained_model`)
  - Accuracy: `0.9847`
  - Train loss: `0.0309`, Epochs: `12`
  - TNR per class: `{'label': 1.0, 'CG': 0.9863, 'OR': 0.9830}`

- **ALBERT-base-v2** (`./alberttrained_model`)
  - Accuracy: `0.9758`
  - Train loss: `0.0649`, Epochs: `12`
  - TNR per class: `{'label': 1.0, 'CG': 0.9876, 'OR': 0.9636}`

- **DeBERTa-v3-base** (`./debertatrained_model`)
  - Accuracy: `0.9817`
  - Train loss: `0.0407`, Epochs: `7`
  - TNR per class: `{'CG': 0.9749, 'OR': 0.9890}`

### Classical Baselines
- **Naive Bayes**
  - Accuracy: `84.01%`
  - TNR: `0.8743`

- **SVM**
  - Accuracy: `85.93%`
  - TNR: `0.8634`

- **KNN**
  - Accuracy: `58.62%`
  - TNR: `0.9588`

- **Random Forest**
  - Accuracy: `83.57%`

- **CNN (custom)**
  - Accuracy: `0.94`
  - TNR: `0.94`

## Notes
- Safe weights: models are loaded using `safetensors` where available to avoid `.bin` loading and related security issues.
- Tokenizers: DeBERTa v3 slow tokenizer requires `sentencepiece`. Fast tokenizer may require `tiktoken` in some environments.
- CSV columns: for UI robustness, upload files should include a recognizable text column (`text` or `text_` recommended) and a `label` column for example sampling.

## License
- For academic/research use. Add your preferred license here if you plan to distribute.

