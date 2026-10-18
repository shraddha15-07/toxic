# 🛡️ ToxiGuard — Toxic Comment Classification App

## Member 5: Streamlit Application & Integration

A premium Streamlit application that integrates all project modules into a unified toxic comment detection and moderation pipeline.

---

## 🚀 Quick Start

### 1. Install Dependencies
```bash
cd nlp
pip install -r requirements.txt
```

### 2. Run the Application
```bash
streamlit run app.py
```

The app will open at `http://localhost:8501`.

---

## 📦 Project Structure

```
nlp/
├── app.py                     # Main Streamlit application (Member 5)
├── predictor.py               # Model loading & inference module
├── explainer.py               # Token-level attention explanations
├── toxic_preprocessing.py     # Shared text cleaning (Member 1)
├── thresholds.json            # Per-label decision thresholds (Member 2)
├── requirements.txt           # Python dependencies
├── README.md                  # This file
│
├── model/                     # Fine-tuned DistilBERT (from Colab — Member 2)
│   ├── config.json
│   ├── model.safetensors
│   └── ...
│
├── tokenizer/                 # Saved tokenizer (from Colab — Member 2)
│   ├── tokenizer_config.json
│   ├── vocab.txt
│   └── ...
│
├── member2_outputs/           # Training outputs (Member 2)
│   ├── test_predictions.parquet
│   ├── val_predictions.parquet
│   ├── thresholds.json
│   └── metrics_test.csv
│
├── member3_outputs/           # Fairness audit (Member 3)
│   ├── group_rates_toxic.csv
│   └── ...
│
├── member4_outputs/           # Calibration + XAI (Member 4)
│   ├── calibration_config.json
│   ├── calibration_before_after.csv
│   └── ...
│
└── processed_data/            # Cleaned data splits (Member 1)
    └── processed_data/
        ├── train.parquet
        ├── val.parquet
        ├── test.parquet
        └── dataset_card.json
```

---

## 🔌 Integration Guide

### Adding the Fine-Tuned Model (from Member 2)

After training on Google Colab, download these files and place them in a `model/` directory:

```bash
# From Google Colab, run:
from google.colab import files
import shutil

# Zip the model directory
shutil.make_archive('/content/model', 'zip', '/content/drive/MyDrive/toxic_project/member2_outputs/model')
files.download('/content/model.zip')

# Also download:
files.download('/content/drive/MyDrive/toxic_project/member2_outputs/thresholds.json')
files.download('/content/drive/MyDrive/toxic_project/member2_outputs/test_predictions.parquet')
```

Then extract locally:
```bash
# Place in the nlp/ directory
unzip model.zip -d model/
cp thresholds.json nlp/
cp test_predictions.parquet nlp/
```

### Adding Fairness Results (from Member 3)
```bash
mkdir member3_outputs
cp group_rates_toxic.csv member3_outputs/
```

### Adding Calibration Config (from Member 4)
```bash
mkdir member4_outputs
cp calibration_config.json member4_outputs/
cp calibration_before_after.csv member4_outputs/
```

---

## 📱 Application Pages

### 1. 🔍 Predict
- Enter any comment and get real-time toxicity predictions
- Shows probability bars with threshold indicators for all 6 labels
- Token-level explanations using attention weights
- Flags uncertain predictions near decision boundaries

### 2. 📈 Model Performance
- Dashboard with per-label Precision, Recall, F1, ROC-AUC
- Interactive bar charts and probability distribution histograms
- Calibration comparison table (if available)
- Falls back to reference metrics if no predictions file is found

### 3. ⚖️ Fairness Audit
- Identity-group FPR and FNR analysis
- Disparity heatmaps showing bias relative to overall rates
- Highlights groups where FPR exceeds 2% above baseline
- Supports all 24 identity categories from the Jigsaw dataset

### 4. 📋 Review Queue
- Tracks all predictions made in the session
- Separates uncertain cases for human review
- Provides Approve/Reject/Escalate actions
- Exportable prediction history

### 5. 📦 Batch Analysis
- Upload CSV files for bulk classification
- Multi-line text input for quick batch processing
- Downloadable results CSV

### 6. ℹ️ About
- System architecture overview
- Technical specifications
- File availability checker

---

## ⚙️ Configuration

### Thresholds
Edit `thresholds.json` to adjust per-label decision boundaries:
```json
{
    "toxic": 0.720,
    "obscene": 0.900,
    "threat": 0.863,
    "insult": 0.789,
    "identity_hate": 0.821,
    "sexual_explicit": 0.920
}
```

### Calibration
Place `calibration_config.json` (from Member 4) for temperature-scaled probabilities:
```json
{
    "method": "temperature",
    "temperatures": {
        "toxic": 1.23,
        "obscene": 0.89,
        ...
    }
}
```

---

## 🧪 Testing

Run the app in demo mode (without the fine-tuned model):
```bash
streamlit run app.py
```

The app will automatically detect that no model is present and use an untrained DistilBERT head. All UI features work in demo mode — you just get random predictions.

To test with the real model, place the model files as described in the Integration Guide above.

---

## 🌐 Deployment

### Local
```bash
streamlit run app.py --server.port 8501
```

### Streamlit Cloud
1. Push the repo to GitHub
2. Go to [share.streamlit.io](https://share.streamlit.io)
3. Connect your repo and set `app.py` as the entry point
4. Add model files to the repo (or use Git LFS for large files)

### Docker
```dockerfile
FROM python:3.10-slim
WORKDIR /app
COPY . .
RUN pip install -r requirements.txt
EXPOSE 8501
CMD ["streamlit", "run", "app.py", "--server.port=8501", "--server.address=0.0.0.0"]
```

```bash
docker build -t toxiguard .
docker run -p 8501:8501 toxiguard
```

---

## 📋 Labels

| Label | Description | Threshold |
|-------|-------------|-----------|
| ☣️ toxic | Rude, disrespectful, or unreasonable | 0.720 |
| 🤬 obscene | Obscene language | 0.900 |
| ⚠️ threat | Threatening language | 0.863 |
| 💢 insult | Insulting language | 0.789 |
| 🚫 identity_hate | Hate based on identity | 0.821 |
| 🔞 sexual_explicit | Sexually explicit content | 0.920 |

---

## 👥 Team

| Member | Role | Key Deliverable |
|--------|------|-----------------|
| 1 | Data Preprocessing | Cleaned dataset, EDA, splits |
| 2 | Model Training | Fine-tuned DistilBERT, thresholds |
| 3 | Fairness Audit | Identity-group bias analysis |
| 4 | XAI & Calibration | SHAP, temperature scaling |
| 5 | Application & Integration | This Streamlit app |
