# Project Requirements

## Purpose

This project develops a review sentiment analysis application using a fine-tuned DistilBERT model. It includes both a Streamlit web app and a notebook for training, comparison, and visualization.

## Recommended Environment

- Operating System: Windows, Linux, or macOS
- Python: 3.10 or 3.11
- RAM: 8 GB minimum (16 GB recommended)
- Optional: GPU for faster transformer training

## Required Python Packages

The main dependencies are:

- streamlit
- pandas
- numpy
- matplotlib
- torch
- transformers
- scikit-learn
- nltk
- wordcloud
- seaborn
- gensim
- jupyterlab
- ipykernel

## Installation

Install all dependencies with:

```bash
pip install -r requirements.txt
```

## Notebook Requirements

For notebook execution, the following extra resources are needed:

- NLTK datasets: punkt, punkt_tab, stopwords
- Internet access for downloading pretrained transformers and NLTK resources

## Running the Project

### Streamlit Web App

```bash
streamlit run appfinal.py
```

### Notebook

Open the notebook at:

```text
notebooks/Model_Selection + BERT.ipynb
```

## Expected Files

- appfinal.py
- dataset.csv
- notebooks/Model_Selection + BERT.ipynb
- models/best_model_bert/
- images/

## Notes

The Streamlit app expects the trained DistilBERT model directory to be available in the models folder. The notebook can be used to reproduce or extend the training pipeline.
