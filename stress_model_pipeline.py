"""
Stress Detection ML Pipeline
Phases 3 & 4: Model Building + Explainability Layer
Handles: Long Reddit text + Class Imbalance
"""

import pandas as pd
import numpy as np
import re
import pickle
import os
import warnings
warnings.filterwarnings('ignore')

# ─── NLP ────────────────────────────────────────────────────────────────────
import nltk
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer

# ─── ML ─────────────────────────────────────────────────────────────────────
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import LinearSVC
from sklearn.model_selection import train_test_split, StratifiedKFold, cross_val_score
from sklearn.metrics import (classification_report, confusion_matrix,
                              accuracy_score, f1_score, roc_auc_score)
from sklearn.pipeline import Pipeline
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import LabelEncoder

# ─── Imbalance fix ──────────────────────────────────────────────────────────
try:
    from imblearn.over_sampling import SMOTE
    from imblearn.pipeline import Pipeline as ImbPipeline
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("⚠  imbalanced-learn not found. Install with: pip install imbalanced-learn")
    print("   Falling back to class_weight='balanced' only.\n")

# ─── Explainability ─────────────────────────────────────────────────────────
try:
    import shap
    SHAP_AVAILABLE = True
except ImportError:
    SHAP_AVAILABLE = False
    print("⚠  SHAP not found. Install with: pip install shap")
    print("   Falling back to feature importance method.\n")


# ════════════════════════════════════════════════════════════════════════════
# STEP 1 — Download NLTK data
# ════════════════════════════════════════════════════════════════════════════
print("Downloading NLTK resources...")
for pkg in ['stopwords', 'wordnet', 'omw-1.4', 'punkt']:
    nltk.download(pkg, quiet=True)

STOP_WORDS = set(stopwords.words('english'))
lemmatizer = WordNetLemmatizer()


# ════════════════════════════════════════════════════════════════════════════
# STEP 2 — Text Preprocessing (Reddit-aware)
# ════════════════════════════════════════════════════════════════════════════
def clean_reddit_text(text: str, max_words: int = 150) -> str:
    """
    Clean Reddit posts:
    - Remove URLs, usernames, subreddit refs, markdown
    - Lowercase, lemmatize, remove stopwords
    - Truncate to max_words (Reddit posts are long — we take the most
      emotionally dense part: first 100 + last 50 words)
    """
    if not isinstance(text, str):
        return ""

    # Remove URLs
    text = re.sub(r'http\S+|www\S+', '', text)
    # Remove Reddit usernames and subreddits
    text = re.sub(r'u/\w+|r/\w+', '', text)
    # Remove markdown (bold, italic, headers, bullets)
    text = re.sub(r'[#*_~`>|]', '', text)
    # Remove special characters, keep letters and spaces
    text = re.sub(r'[^a-zA-Z\s]', ' ', text)
    # Lowercase
    text = text.lower()
    # Tokenize
    tokens = text.split()

    # ── Smart truncation for long posts ──────────────────────────────────
    # Take first 100 + last 50 words → captures intro (context) + conclusion
    # (where people often reveal their emotional state)
    if len(tokens) > max_words:
        tokens = tokens[:100] + tokens[-50:]

    # Remove stopwords + lemmatize
    tokens = [
        lemmatizer.lemmatize(t)
        for t in tokens
        if t not in STOP_WORDS and len(t) > 2
    ]

    return ' '.join(tokens)


# ════════════════════════════════════════════════════════════════════════════
# STEP 3 — Load & Prepare Dataset
# ════════════════════════════════════════════════════════════════════════════
def load_dataset(filepath: str) -> pd.DataFrame:
    """
    Load your dataset. Expects columns: 'text' and 'label'.
    Label should be binary: 1 = stress/high, 0 = no stress/low.

    Adjust column names below to match your actual CSV.
    """
    df = pd.read_csv(filepath)

    # ── Adapt these to your actual column names ───────────────────────────
    # Common Reddit dataset column names:
    possible_text_cols  = ['text', 'post', 'body', 'content', 'selftext', 'message']
    possible_label_cols = ['label', 'stress_label', 'category', 'class', 'target']

    text_col  = next((c for c in possible_text_cols  if c in df.columns), None)
    label_col = next((c for c in possible_label_cols if c in df.columns), None)

    if not text_col or not label_col:
        print(f"Available columns: {list(df.columns)}")
        raise ValueError(
            "Could not auto-detect text/label columns. "
            "Please update `possible_text_cols` / `possible_label_cols` above."
        )

    df = df[[text_col, label_col]].rename(columns={text_col: 'text', label_col: 'label'})
    df.dropna(subset=['text', 'label'], inplace=True)

    # ── Normalize labels to 0/1 ──────────────────────────────────────────
    # If labels are already 0/1 integers, this is a no-op.
    # If labels are strings like 'stress'/'no stress', encode them.
    unique_labels = df['label'].unique()
    if not set(unique_labels).issubset({0, 1}):
        le = LabelEncoder()
        df['label'] = le.fit_transform(df['label'])
        print(f"Label encoding: {dict(zip(le.classes_, le.transform(le.classes_)))}")

    return df


def show_class_distribution(df: pd.DataFrame):
    counts = df['label'].value_counts()
    total  = len(df)
    print("\n📊 Class Distribution:")
    print(f"   Class 0 (Low Stress) : {counts.get(0, 0):>6}  ({counts.get(0,0)/total*100:.1f}%)")
    print(f"   Class 1 (High Stress): {counts.get(1, 0):>6}  ({counts.get(1,0)/total*100:.1f}%)")
    ratio = counts.max() / counts.min() if counts.min() > 0 else float('inf')
    if ratio > 2:
        print(f"   ⚠  Imbalance ratio {ratio:.1f}:1 — applying SMOTE + class_weight fix")
    print()


# ════════════════════════════════════════════════════════════════════════════
# STEP 4 — Build Models (class-imbalance aware)
# ════════════════════════════════════════════════════════════════════════════
def build_models(use_smote: bool = True) -> dict:
    """
    Returns dict of model pipelines.
    All classifiers use class_weight='balanced' as the first line of defense.
    If SMOTE is available, it's added as an extra layer.
    """
    tfidf = TfidfVectorizer(
        max_features=10_000,   # enough for Reddit vocabulary
        ngram_range=(1, 2),    # unigrams + bigrams catch phrases like "can't cope"
        sublinear_tf=True,     # log-scale TF — helps with long documents
        min_df=3,              # ignore very rare words (typos, noise)
        max_df=0.90,           # ignore words that appear in >90% of posts
    )

    lr  = LogisticRegression(class_weight='balanced', max_iter=1000, C=1.0, random_state=42)
    rf  = RandomForestClassifier(class_weight='balanced', n_estimators=200,
                                  max_depth=20, random_state=42, n_jobs=-1)
    # LinearSVC needs calibration for predict_proba
    svc_raw = LinearSVC(class_weight='balanced', max_iter=2000, random_state=42)
    svc = CalibratedClassifierCV(svc_raw)

    models = {}

    if use_smote and SMOTE_AVAILABLE:
        # SMOTE works on the vectorized features, so: vectorize → SMOTE → classify
        smote = SMOTE(random_state=42, k_neighbors=3)
        models['Logistic Regression + SMOTE'] = ImbPipeline([
            ('tfidf', tfidf), ('smote', smote), ('clf', lr)
        ])
        models['Random Forest + SMOTE'] = ImbPipeline([
            ('tfidf', tfidf), ('smote', smote), ('clf', rf)
        ])
        models['SVM + SMOTE'] = ImbPipeline([
            ('tfidf', tfidf), ('smote', smote), ('clf', svc)
        ])
    else:
        # Fall back to sklearn Pipeline (class_weight='balanced' still applied)
        from sklearn.pipeline import Pipeline as SkPipeline
        models['Logistic Regression'] = SkPipeline([('tfidf', tfidf), ('clf', lr)])
        models['Random Forest']        = SkPipeline([('tfidf', tfidf), ('clf', rf)])
        models['SVM']                  = SkPipeline([('tfidf', tfidf), ('clf', svc)])

    return models


# ════════════════════════════════════════════════════════════════════════════
# STEP 5 — Train & Evaluate
# ════════════════════════════════════════════════════════════════════════════
def train_and_evaluate(df: pd.DataFrame) -> tuple:
    """
    Trains all models, prints evaluation, returns best pipeline + vectorizer.
    Uses Stratified Split to maintain class ratio in train/test.
    """
    X = df['text'].values
    y = df['label'].values

    # Stratified split — preserves class ratio in both sets
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    print(f"Train size: {len(X_train)} | Test size: {len(X_test)}\n")

    models = build_models(use_smote=SMOTE_AVAILABLE)

    results = {}
    for name, pipeline in models.items():
        print(f"🔧 Training: {name}")
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        acc  = accuracy_score(y_test, y_pred)
        f1   = f1_score(y_test, y_pred, average='weighted')
        auc  = roc_auc_score(y_test, y_prob)

        results[name] = {'pipeline': pipeline, 'f1': f1, 'auc': auc, 'acc': acc}

        print(f"   Accuracy : {acc:.4f}")
        print(f"   F1 Score : {f1:.4f}  ← main metric (handles imbalance)")
        print(f"   ROC-AUC  : {auc:.4f}")
        print(f"\n{classification_report(y_test, y_pred, target_names=['Low Stress','High Stress'])}")
        print("-" * 60)

    # Pick best by F1 (not accuracy — accuracy is misleading for imbalanced data)
    best_name = max(results, key=lambda k: results[k]['f1'])
    best      = results[best_name]
    print(f"\n✅ Best Model: {best_name}")
    print(f"   F1={best['f1']:.4f} | AUC={best['auc']:.4f} | Acc={best['acc']:.4f}")

    return best['pipeline'], best_name, X_test, y_test


# ════════════════════════════════════════════════════════════════════════════
# STEP 6 — Explainability Layer (SHAP or fallback)
# ════════════════════════════════════════════════════════════════════════════
def get_top_words_shap(pipeline, text: str, n: int = 5) -> list[dict]:
    """
    Returns top N words that influenced the prediction using SHAP.
    Works with Logistic Regression pipelines.
    """
    tfidf = pipeline.named_steps['tfidf']
    clf   = pipeline.named_steps['clf']

    # Transform single text
    X_vec = tfidf.transform([text])

    explainer = shap.LinearExplainer(clf, X_vec, feature_perturbation="interventional")
    shap_vals  = explainer.shap_values(X_vec)

    # shap_vals shape: (1, n_features) for binary
    if isinstance(shap_vals, list):
        vals = shap_vals[1][0]   # class 1 (stress)
    else:
        vals = shap_vals[0]

    feature_names = tfidf.get_feature_names_out()
    word_shap     = list(zip(feature_names, vals))

    # Filter to words actually present in the input
    present = set(text.lower().split())
    word_shap = [(w, v) for w, v in word_shap if w in present]

    # Sort by absolute impact
    word_shap.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            'word': w,
            'impact': float(v),
            'direction': 'increases stress' if v > 0 else 'reduces stress'
        }
        for w, v in word_shap[:n]
    ]


def get_top_words_coef(pipeline, text: str, n: int = 5) -> list[dict]:
    """
    Fallback explainability using logistic regression coefficients.
    Works when SHAP is not installed.
    """
    tfidf = pipeline.named_steps['tfidf']
    clf   = pipeline.named_steps['clf']

    feature_names = tfidf.get_feature_names_out()

    # Get coefficients (works for LR; for RF use feature_importances_)
    if hasattr(clf, 'coef_'):
        coefs = clf.coef_[0]
    elif hasattr(clf, 'feature_importances_'):
        coefs = clf.feature_importances_
    else:
        return []

    word_to_coef = dict(zip(feature_names, coefs))

    cleaned_text = clean_reddit_text(text, max_words=200)
    present_words = set(cleaned_text.split())

    scored = [
        (w, word_to_coef[w])
        for w in present_words
        if w in word_to_coef
    ]
    scored.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            'word': w,
            'impact': float(v),
            'direction': 'increases stress' if v > 0 else 'reduces stress'
        }
        for w, v in scored[:n]
    ]


def explain_prediction(pipeline, raw_text: str, n: int = 5) -> dict:
    """
    Full prediction + explanation for a single input text.
    Returns:
        label       : 'HIGH STRESS' or 'LOW STRESS'
        confidence  : float 0–1
        top_words   : list of {word, impact, direction}
        explanation : human-readable string
    """
    cleaned = clean_reddit_text(raw_text)

    pred    = pipeline.predict([cleaned])[0]
    prob    = pipeline.predict_proba([cleaned])[0]
    conf    = float(prob[pred])
    label   = 'HIGH STRESS' if pred == 1 else 'LOW STRESS'

    # Choose explainability method
    try:
        if SHAP_AVAILABLE and 'LogisticRegression' in type(pipeline.named_steps['clf']).__name__:
            top_words = get_top_words_shap(pipeline, cleaned, n)
        else:
            top_words = get_top_words_coef(pipeline, cleaned, n)
    except Exception as e:
        print(f"Explainability warning: {e}")
        top_words = get_top_words_coef(pipeline, cleaned, n)

    # Build human-readable explanation
    stress_words    = [w['word'] for w in top_words if w['direction'] == 'increases stress']
    nonstress_words = [w['word'] for w in top_words if w['direction'] == 'reduces stress']

    if stress_words:
        explanation = f"Words like \"{', '.join(stress_words[:3])}\" strongly indicate stress."
    elif nonstress_words:
        explanation = f"Words like \"{', '.join(nonstress_words[:3])}\" suggest a calm tone."
    else:
        explanation = "Based on the overall pattern of your text."

    return {
        'label'      : label,
        'confidence' : round(conf * 100, 1),
        'top_words'  : top_words,
        'explanation': explanation,
    }


# ════════════════════════════════════════════════════════════════════════════
# STEP 7 — Save Model
# ════════════════════════════════════════════════════════════════════════════
def save_model(pipeline, path: str = 'stress_model.pkl'):
    with open(path, 'wb') as f:
        pickle.dump(pipeline, f)
    print(f"\n💾 Model saved to: {path}")


def load_model(path: str = 'stress_model.pkl'):
    with open(path, 'rb') as f:
        return pickle.load(f)


# ════════════════════════════════════════════════════════════════════════════
# MAIN — Run Everything
# ════════════════════════════════════════════════════════════════════════════
if __name__ == '__main__':
    import sys

    # ── 1. Load Dataset ───────────────────────────────────────────────────
    dataset_path = sys.argv[1] if len(sys.argv) > 1 else 'dataset.csv'

    if not os.path.exists(dataset_path):
        print(f"❌ Dataset not found at: {dataset_path}")
        print("   Usage: python stress_model_pipeline.py path/to/your_dataset.csv")
        sys.exit(1)

    print(f"📂 Loading dataset: {dataset_path}")
    df = load_dataset(dataset_path)
    show_class_distribution(df)

    # ── 2. Clean Text ─────────────────────────────────────────────────────
    print("🧹 Cleaning text (Reddit-aware)...")
    df['text'] = df['text'].apply(clean_reddit_text)
    df = df[df['text'].str.strip().astype(bool)]   # drop empty rows after cleaning
    print(f"   {len(df)} samples after cleaning\n")

    # ── 3. Train & Evaluate ───────────────────────────────────────────────
    best_pipeline, best_name, X_test, y_test = train_and_evaluate(df)

    # ── 4. Save Model ─────────────────────────────────────────────────────
    save_model(best_pipeline, 'stress_model.pkl')

    # ── 5. Quick Demo ─────────────────────────────────────────────────────
    print("\n" + "="*60)
    print("🔍 DEMO — Explainability Test")
    print("="*60)

    test_inputs = [
        "I can't sleep, I feel so overwhelmed and hopeless. Nothing is working out.",
        "Had a great walk today, feeling calm and grateful for everything.",
        "Deadlines are piling up, I'm exhausted and I don't know what to do anymore."
    ]

    for text in test_inputs:
        result = explain_prediction(best_pipeline, text)
        print(f"\nInput   : {text[:80]}...")
        print(f"Result  : {result['label']}  ({result['confidence']}% confident)")
        print(f"Reason  : {result['explanation']}")
        if result['top_words']:
            print("Top words:")
            for w in result['top_words']:
                sign = '🔴' if w['direction'] == 'increases stress' else '🟢'
                print(f"   {sign} '{w['word']}' — {w['direction']} (score: {w['impact']:.3f})")