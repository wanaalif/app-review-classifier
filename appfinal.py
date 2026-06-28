import os

import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from transformers import DistilBertForSequenceClassification, DistilBertTokenizerFast

try:
    from huggingface_hub import snapshot_download
except ImportError:  # pragma: no cover
    snapshot_download = None

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------

st.set_page_config(
    page_title="Sentiment Analysis App",
    page_icon="💬",
    layout="wide"
)

# --------------------------------------------------
# LOAD MODEL (DistilBERT)
# --------------------------------------------------

def get_setting(name, default=None):
    value = os.environ.get(name)
    if value:
        return value

    try:
        secret_value = st.secrets.get(name)
    except Exception:
        secret_value = None

    if secret_value is not None:
        return secret_value

    return default


MODEL_PATH = get_setting("MODEL_PATH", "models/best_model_bert")
MODEL_REPO_ID = get_setting("MODEL_REPO_ID")
MODEL_REVISION = get_setting("MODEL_REVISION", "main")

# Label order matches sklearn LabelEncoder's alphabetical sort of
# ['negative', 'neutral', 'positive'] used during training.
LABEL_MAP = {0: "Negative", 1: "Neutral", 2: "Positive"}


def ensure_model_available():
    if os.path.isdir(MODEL_PATH) and os.path.exists(os.path.join(MODEL_PATH, "config.json")):
        return MODEL_PATH

    if MODEL_REPO_ID:
        if snapshot_download is None:
            raise RuntimeError("huggingface_hub is required to download the model from Hugging Face Hub.")

        os.makedirs("models", exist_ok=True)
        st.write("Local model files were not found. Downloading from Hugging Face Hub...")
        try:
            return snapshot_download(
                repo_id=MODEL_REPO_ID,
                revision=MODEL_REVISION,
                local_dir=MODEL_PATH,
                local_dir_use_symlinks=False,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Could not download model from Hugging Face Hub: {exc}"
            ) from exc

    raise FileNotFoundError(
        f"Model directory '{MODEL_PATH}' was not found. "
        "Set MODEL_REPO_ID to a public Hugging Face model repo for Streamlit Cloud deployment."
    )


@st.cache_resource
def load_model():
    resolved_model_path = ensure_model_available()
    model = DistilBertForSequenceClassification.from_pretrained(resolved_model_path)
    tokenizer = DistilBertTokenizerFast.from_pretrained(resolved_model_path)
    model.eval()
    return model, tokenizer


try:
    model, tokenizer = load_model()
except Exception as exc:
    st.error(f"Model initialization failed: {exc}")
    model = None
    tokenizer = None


@st.cache_data
def load_data():
    return pd.read_csv("dataset.csv")


df = load_data()

# --------------------------------------------------
# SIDEBAR
# --------------------------------------------------

st.sidebar.title("Navigation")

page = st.sidebar.radio(
    "Go to",
    [
        "🏠 Home/About",
        "📝 Text Analyzer",
        "📊 Data Explorer",
        "📈 Visualizations",
        "🤖 Model Info"
    ]
)

# ==================================================
# HOME PAGE
# ==================================================

if page == "🏠 Home/About":

    st.title("💬 Review Sentiment Analysis")

    st.header("Project Description")
    st.write("""
    This application predicts whether a review is Positive, Negative, or Neutral
    using a fine-tuned DistilBERT transformer model.
    """)

    st.header("Problem Statement")
    st.write("""
    Manually analyzing thousands of reviews is time-consuming.
    This system automatically classifies customer reviews.
    """)

    st.header("How to Use")
    st.write("""
    1. Navigate to Text Analyzer.
    2. Enter a review.
    3. Click Analyze.
    4. View the prediction results.
    """)

    st.header("Team Members")
    st.write("""
    - Alif
    - Rania
    - Kanesh
    - Nurien
    """)

# ==================================================
# TEXT ANALYZER
# ==================================================

elif page == "📝 Text Analyzer":

    st.title("📝 Text Analyzer")

    review = st.text_area(
        "Enter your review:",
        height=150
    )

    if st.button("Analyze"):

        if review.strip() == "":
            st.warning("Please enter a review.")
        else:
            # Tokenize raw text (BERT was fine-tuned on raw text, not cleaned text)
            inputs = tokenizer(
                review,
                return_tensors="pt",
                truncation=True,
                padding=True,
                max_length=256
            )

            with torch.no_grad():
                outputs = model(**inputs)

            probs = F.softmax(outputs.logits, dim=1).numpy()[0]
            prediction = int(np.argmax(probs))
            confidence = float(np.max(probs)) * 100
            label = LABEL_MAP[prediction]

            st.subheader("Prediction Result")

            if label == "Positive":
                st.success("Positive Review 😊")
            elif label == "Negative":
                st.error("Negative Review 😠")
            else:
                st.info("Neutral Review 😐")

            st.subheader("Confidence Score")

            st.progress(float(confidence / 100))

            st.write(f"Confidence: {confidence:.2f}%")

            # Show probability breakdown across all classes
            st.subheader("Class Probabilities")

            prob_df = pd.DataFrame({
                "Sentiment": [LABEL_MAP[i] for i in range(len(probs))],
                "Probability": probs
            }).set_index("Sentiment")

            st.bar_chart(prob_df)

            # ----------------------------------------------------
            # Words Influencing Prediction (leave-one-word-out)
            # ----------------------------------------------------
            st.subheader("Words Influencing Prediction")
            st.caption(
                "Each word is temporarily removed from the review, and the model's "
                "confidence drop for the predicted class shows how much that word "
                "mattered. Bigger drop = more influential word."
            )

            words = review.split()

            if len(words) <= 1:
                st.write("Enter a longer review to see word-level influence.")
            else:
                influences = []
                for i in range(len(words)):
                    # Remove word i and re-predict
                    masked_text = " ".join(words[:i] + words[i + 1:])
                    masked_inputs = tokenizer(
                        masked_text,
                        return_tensors="pt",
                        truncation=True,
                        padding=True,
                        max_length=256
                    )
                    with torch.no_grad():
                        masked_outputs = model(**masked_inputs)
                    masked_probs = F.softmax(masked_outputs.logits, dim=1).numpy()[0]

                    # How much did removing this word drop confidence in the predicted class?
                    drop = confidence - (masked_probs[prediction] * 100)
                    influences.append(drop)

                influence_df = pd.DataFrame({
                    "Word": words,
                    "Influence": influences
                }).sort_values("Influence", ascending=False)

                st.dataframe(influence_df, use_container_width=True, hide_index=True)

                fig, ax = plt.subplots(figsize=(8, max(2, len(words) * 0.3)))
                plot_df = influence_df.sort_values("Influence")
                colors = ["#e74c3c" if v < 0 else "#2ecc71" for v in plot_df["Influence"]]
                ax.barh(plot_df["Word"], plot_df["Influence"], color=colors)
                ax.set_xlabel("Confidence Drop When Removed (%)")
                ax.set_title("Word Influence on Prediction")
                st.pyplot(fig)

# ==================================================
# DATA EXPLORER
# ==================================================

elif page == "📊 Data Explorer":

    st.title("📊 Data Explorer")

    st.subheader("Sample Dataset")

    st.dataframe(df.head(20))

    st.subheader("Dataset Statistics")

    st.write("Number of Rows:", df.shape[0])
    st.write("Number of Columns:", df.shape[1])

    st.write(df.describe())

    st.subheader("Missing Values")

    st.write(df.isnull().sum())

# ==================================================
# VISUALIZATION PAGE
# ==================================================

elif page == "📈 Visualizations":

    st.title("📈 Visualizations")

    # Label Distribution (from saved notebook image)
    st.subheader("Sentiment Class Distribution")
    try:
        st.image("images/label_distribution.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/label_distribution.png")

    # Word Cloud
    st.subheader("Word Cloud")
    try:
        st.image("images/wc_new.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/wc_new.png")

    # Model Comparison Chart
    st.subheader("Model Comparison Chart")
    try:
        st.image("images/model_comparison_chart.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/model_comparison_chart.png")

    # Accuracy Comparison
    st.subheader("Accuracy Comparison Across Models")
    try:
        st.image("images/viz_accuracy_comparison.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/viz_accuracy_comparison.png")

    # F1 Score Per Class
    st.subheader("F1 Score Per Class")
    try:
        st.image("images/viz_f1_per_class.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/viz_f1_per_class.png")

    # Confusion Matrix
    st.subheader("Confusion Matrices")
    try:
        st.image("images/viz_confusion_matrices.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/viz_confusion_matrices.png")

    # ROC Curve
    st.subheader("ROC Curves")
    try:
        st.image("images/viz_roc_curves.png", use_container_width=True)
    except Exception:
        st.warning("Please add images/viz_roc_curves.png")

# ==================================================
# MODEL INFO
# ==================================================

elif page == "🤖 Model Info":

    st.title("🤖 Model Information")

    st.subheader("Models Explored")

    st.write("""
    Three different modeling approaches were trained and compared for this
    sentiment classification task:

    - **Config A — Logistic Regression × TF-IDF:** A linear baseline using
      sparse word-frequency features. SMOTE was applied to balance the
      training data before fitting, and the best regularization strength
      (C) was found via GridSearchCV (3-fold Stratified CV, optimized on
      f1_macro).
    - **Config B — Logistic Regression × Word2Vec:** Same linear model, but
      using dense, sentence-averaged word embeddings instead of TF-IDF.
      Also trained on SMOTE-balanced data with GridSearchCV tuning.
    - **Config C — DistilBERT (Fine-Tuned Transformer):** A deep contextual
      model fine-tuned end-to-end on raw, imbalanced text. It uses
      self-attention to learn context directly, with no manual feature
      engineering or oversampling.
    """)

    st.subheader("Comparison of All Configurations")

    comparison_df = pd.DataFrame({
        "Configuration": ["LR × TF-IDF", "LR × Word2Vec", "DistilBERT"],
        "Accuracy": [0.7681, 0.7395, 0.8964],
        "Precision": [0.6130, 0.5944, 0.7492],
        "Recall": [0.6436, 0.6188, 0.6599],
        "F1-Macro": [0.6042, 0.5795, 0.6781],
        "F1-Weighted": [0.8032, 0.7819, 0.8846],
    }).set_index("Configuration")

    st.dataframe(comparison_df.style.format("{:.4f}"), use_container_width=True)

    st.success(
        "🏆 **DistilBERT** was selected as the deployed model — it achieved "
        "the highest F1-Macro (0.6781), meaning it performs best across all "
        "three sentiment classes (Positive, Negative, and Neutral), not just "
        "the majority class."
    )

    st.subheader("Deployed Model: DistilBERT")

    st.write("""
    - Algorithm: DistilBERT (Fine-Tuned Transformer)
    - Feature Extraction: Native tokenization (no manual feature engineering —
      self-attention captures context directly from raw text)
    - Imbalance Handling: None — trained on raw imbalanced data to preserve
      the transformer's pre-trained language priors
    - Training: Fine-tuned for 2 epochs
    - Training Loss: 0.4142 (Epoch 1) → 0.3339 (Epoch 2)
    """)

    st.subheader("Deployed Model Performance Metrics")

    st.write("""
    Accuracy: 89.64%

    Precision: 74.92%

    Recall: 65.99%

    F1-Macro: 67.81%

    F1-Weighted: 88.46%
    """)

    st.caption(
        "Note: F1-Macro is notably lower than F1-Weighted because the Neutral "
        "class remains a challenging minority class (F1-score of only 0.23) "
        "even for the best-performing model."
    )
