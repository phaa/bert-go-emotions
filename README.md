# BERT GoEmotions: Multi-label Emotion Classification

## Overview

This project applies state-of-the-art NLP techniques to classify multiple emotions in short texts using Google's [GoEmotions](https://research.google/blog/goemotions-a-dataset-for-fine-grained-emotion-classification/) dataset. The core model is BERT, fine-tuned with PyTorch and Hugging Face Transformers for multi-label emotion detection.

## Objectives

- Fine-tune a pre-trained BERT model (`bert-base-uncased`) for emotion classification.
- Handle multi-label classification (28 emotion classes).
- Export and deploy the trained model with FastAPI for inference.

## Dataset

- **Source:** [GoEmotions](https://huggingface.co/datasets/go_emotions)
- **Description:** 58,000+ English Reddit comments, each annotated with one or more of 28 emotion labels.
- **Classes:** admiration, amusement, anger, annoyance, approval, caring, confusion, curiosity, desire, disappointment, disapproval, disgust, embarrassment, excitement, fear, gratitude, grief, joy, love, nervousness, optimism, pride, realization, relief, remorse, sadness, surprise, neutral.

## Project Structure

- `index.ipynb`: Main notebook with all code for data loading, preprocessing, model training, evaluation, and export.
- `goemotions-bert/`: Directory where the fine-tuned model and tokenizer are saved after training.

## Methodology

1. **Library Installation:**  
   Installs PyTorch, Transformers, Datasets, and Scikit-learn.

2. **Data Loading & Exploration:**  
   Loads the GoEmotions dataset using Hugging Face Datasets. Analyzes label distribution and checks for class imbalance.

3. **Preprocessing:**  
   - Multi-hot encoding for multi-label targets.
   - Tokenization using BERT tokenizer.
   - Conversion of labels to float tensors for compatibility with `BCEWithLogitsLoss`.

4. **Model Definition:**  
   - Loads `BertForSequenceClassification` with `problem_type="multi_label_classification"`.
   - Moves model to GPU if available.

5. **Training:**  
   - Configures hyperparameters (learning rate, batch size, epochs, weight decay).
   - Uses Hugging Face `Trainer` for training and evaluation.
   - Computes F1-score and accuracy for multi-label predictions.

6. **Saving the Model:**  
   Saves the fine-tuned model and tokenizer for later use.

7. **Inference:**  
   - Uses Hugging Face `pipeline` for multi-label emotion prediction on new texts.
   - Maps predicted label indices to emotion names.

8. **Next Steps:**  
   - Deploy the model with FastAPI to serve predictions via HTTP API.

## Dependencies

- `torch`
- `transformers`
- `datasets`
- `scikit-learn`
- `matplotlib`

These dependencies are installed at the beginning of the notebook.

## How to Run

## How to Use  
1. Clone and access the repository:  
   ```bash
   git clone https://github.com/phaa/bert-go-emotions.git
   cd bert-go-emotions/
   ```
2. Activate the virtual environment (conda or venv):
   ```bash
   conda activate ibmenv
   ```
3. Run the notebooks in Jupyter lab:  
   ```bash
   jupyter lab
   ```
*The notebook has a cell to install the necessary dependencies.* 

## Results and Analysis
- The fine-tuned BERT model achieves multi-label emotion classification on the GoEmotions dataset.
- Evaluation metrics (F1, accuracy) are computed and displayed in the notebook.
- Example inference is shown for custom input texts.

## Author

[Pedro Henrique Amorim de Azevedo](https://www.linkedin.com/in/pedro-henrique-amorim-de-azevedo/)
