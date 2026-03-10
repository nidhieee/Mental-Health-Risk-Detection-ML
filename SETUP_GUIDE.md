# Stress Detector — Setup Guide

## Files You Receive
```
stress_model_pipeline.py   ← Phase 3 + 4 (model training + explainability)
app.py                     ← Phase 6 (Streamlit UI)
```

---

## Step 1 — Install Dependencies

```bash
pip install scikit-learn pandas numpy nltk shap imbalanced-learn streamlit
```

---

## Step 2 — Train the Model

Place your dataset CSV in the same folder, then run:

```bash
python stress_model_pipeline.py your_dataset.csv
```

### What your CSV must have:
| Column | What it contains |
|--------|-----------------|
| `text` | The Reddit post / journal entry |
| `label` | `1` = stress, `0` = no stress (or string labels — auto-encoded) |

> Common Reddit dataset column names are auto-detected. If it fails, the script prints your columns so you can update the names.

This will:
- Clean and preprocess all text (Reddit-aware)
- Train 3 models (Logistic Regression, Random Forest, SVM)
- Fix class imbalance with SMOTE + `class_weight='balanced'`
- Print F1, AUC, accuracy for each model
- Save the best model as `stress_model.pkl`

---

## Step 3 — Run the App

```bash
streamlit run app.py
```

Open `http://localhost:8501` in your browser.

---

## How the Two Problems Are Fixed

### Problem 1: Long Reddit Posts
- URLs, usernames (`u/`, `r/`), markdown stripped first
- Smart truncation: takes **first 100 words + last 50 words**
  - First 100 = context/background
  - Last 50 = where people reveal their emotional state
- TF-IDF uses `sublinear_tf=True` (log scaling) — prevents long posts from dominating

### Problem 2: Class Imbalance (more stress posts)
- **SMOTE** (Synthetic Minority Oversampling) — creates synthetic non-stress examples
- **`class_weight='balanced'`** on all classifiers — penalizes misclassifying the minority class more
- **Stratified train/test split** — ensures both sets have the same class ratio
- **F1 Score used for model selection** (not accuracy — accuracy is misleading for imbalanced data)

---

## Output Format (from `explain_prediction`)

```python
{
  'label'      : 'HIGH STRESS',          # or 'LOW STRESS'
  'confidence' : 87.3,                    # percentage
  'top_words'  : [
      {'word': 'hopeless', 'impact': 0.82, 'direction': 'increases stress'},
      {'word': 'calm',     'impact': -0.41,'direction': 'reduces stress'},
  ],
  'explanation': 'Words like "hopeless", "exhausted" strongly indicate stress.'
}
```

---

## Ethical Note (for your report)
> This tool is an early-awareness screening aid, **not a medical diagnosis tool**.  
> Always include helpline information for high-risk outputs.  
> iCall (India): 9152987821 | Vandrevala Foundation: 1860-2662-345
