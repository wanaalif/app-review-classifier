# Sentiment Analysis App

This repository contains a Streamlit web application and a Jupyter notebook for sentiment analysis of customer reviews. The deployed model is a fine-tuned DistilBERT classifier that predicts whether a review is Negative, Neutral, or Positive.

## Project Overview

The project was built to automate review classification and make it easier to analyze large volumes of customer feedback. It combines:

- a web interface for live predictions,
- a notebook for model comparison and training,
- and a set of visualizations for exploratory analysis.

## Features

- Text analysis with instant sentiment prediction
- Confidence scores and class probability breakdowns
- Word-level influence analysis for individual reviews
- Dataset exploration and summary statistics
- Visualization pages for class distribution, word clouds, and model comparison
- A trained DistilBERT model ready for inference

## Project Structure

- appfinal.py — Streamlit app entry point
- notebooks/Model_Selection + BERT.ipynb — notebook with model training and comparison
- dataset.csv — processed dataset used by the app
- models/best_model_bert/ — trained DistilBERT model and tokenizer
- images/ — visual assets used in the app

## Requirements

Python 3.10 or newer is recommended.

Install the dependencies:

```bash
pip install -r requirements.txt
```

If you are using the notebook for the first time, also download the required NLTK data:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

## Run the Streamlit App

From the project root, run:

```bash
streamlit run appfinal.py
```

## Reproduce the Notebook

Open the notebook in [notebooks/Model_Selection + BERT.ipynb](notebooks/Model_Selection%20+%20BERT.ipynb) to review the training workflow, model comparison, and visualization steps.

If you want to rebuild the dataset from raw source files, place the source archive as archive.zip in the project root before running the notebook cells.

## Model Details

The deployed model is a fine-tuned DistilBERT classifier trained for review sentiment classification. The app loads the model directly from the saved model directory.

## Notes

- The app uses the saved model artifacts in the models directory.
- The notebook contains additional experiments for baseline models such as logistic regression with TF-IDF and Word2Vec.
- The project is designed for local use and can be published to GitHub as-is.