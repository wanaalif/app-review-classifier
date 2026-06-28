# SentiAI · Sentiment Analysis

SentiAI is a Streamlit web application for classifying customer reviews into Negative, Neutral, or Positive sentiment using a fine-tuned DistilBERT model. The app combines live inference with exploration views for the dataset and model results.

## What the project does

This project was built to make review analysis faster and more consistent. It provides:

- live sentiment prediction from user-entered text,
- confidence scores and class probability breakdowns,
- word-level influence analysis for individual reviews,
- dataset exploration and summary statistics,
- visualization pages for model and data insights.

## Project structure

- [appfinal.py](appfinal.py) — main Streamlit app entry point
- [requirements.txt](requirements.txt) — Python dependencies for local and cloud deployment
- [dataset.csv](dataset.csv) — review dataset used by the app
- [models](models) — optional local folder for a downloaded model; the deployed model is hosted on Hugging Face Hub
- [notebooks/Model_Selection + BERT.ipynb](notebooks/Model_Selection%20+%20BERT.ipynb) — training and model comparison notebook
- [images](images) — visualization assets used in the app

## Tech stack

- Python
- Streamlit
- pandas
- NumPy
- Matplotlib
- PyTorch
- Hugging Face Transformers
- scikit-learn

## Setup locally

Python 3.10 or 3.11 is recommended.

1. Clone the repository.
2. Install dependencies:

```bash
pip install -r requirements.txt
```

3. Download the required NLTK data if needed:

```python
import nltk
nltk.download("punkt")
nltk.download("punkt_tab")
nltk.download("stopwords")
```

4. Run the app:

```bash
streamlit run appfinal.py
```

## Model loading

The app first tries to load a local model folder if one is present under [models](models). If that folder is unavailable, it downloads the model from a public Hugging Face Hub repository using the `MODEL_REPO_ID` setting.

The trained model is available from Hugging Face at [wanaalif/Sentiment-Analysis-App-Review-Classifier](https://huggingface.co/wanaalif/Sentiment-Analysis-App-Review-Classifier).

For Streamlit Cloud deployment, add the following secrets:

- `MODEL_REPO_ID` — your Hugging Face repository name, for example `wanaalif/Sentiment-Analysis-App-Review-Classifier`
- `MODEL_REVISION` — optional, usually `main`

The model repository should contain the standard Hugging Face files such as `config.json`, `tokenizer_config.json`, `tokenizer.json`, and `model.safetensors` at the repository root.

## Deployment notes

The project is also set up for Streamlit Cloud deployment. A compatible Python runtime pin is included in the repository files so the environment uses a supported version for the Transformers stack.

## Notebook and experiments

The notebook in [notebooks/Model_Selection + BERT.ipynb](notebooks/Model_Selection%20+%20BERT.ipynb) contains the training workflow, model comparison steps, and additional experiments with baseline methods such as logistic regression with TF-IDF and Word2Vec.

## Notes

- The app is designed for local development and for deployment on Streamlit Cloud.
- The model artifacts are large, so hosting them on Hugging Face Hub is the most practical option for cloud deployment.
- The app includes a small compatibility setting for Windows environments to avoid a common PyTorch runtime issue.