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
# STEP 6 — Explainability Layer
# ════════════════════════════════════════════════════════════════════════════

def _get_clf_coefs(clf):
    """
    Safely extract per-feature coefficients from any classifier type,
    including calibrated wrappers (CalibratedClassifierCV).
    Returns array of shape (n_features,) — positive = stress signal.
    """
    # Unwrap CalibratedClassifierCV
    if hasattr(clf, 'calibrated_classifiers_'):
        inner = clf.calibrated_classifiers_[0].estimator
        if hasattr(inner, 'coef_'):
            return inner.coef_[0]
        if hasattr(inner, 'feature_importances_'):
            return inner.feature_importances_

    if hasattr(clf, 'coef_'):
        return clf.coef_[0]

    if hasattr(clf, 'feature_importances_'):
        return clf.feature_importances_

    return None


def get_top_words(pipeline, cleaned_text: str, n: int = 5) -> list[dict]:
    """
    Core explainability: finds which words/phrases from the input most
    influenced the prediction.

    Strategy:
    1. Transform the cleaned text through TF-IDF (same as during training).
    2. Get the feature vector — only non-zero entries are features PRESENT
       in this input. This is the correct way to find active features.
    3. Multiply each active feature's TF-IDF weight by its model coefficient.
       → This gives the actual contribution of each word to the score.
    4. Sort by absolute contribution to find the most impactful words.
    """
    tfidf = pipeline.named_steps['tfidf']
    clf   = pipeline.named_steps['clf']

    coefs = _get_clf_coefs(clf)
    if coefs is None:
        return []

    feature_names = tfidf.get_feature_names_out()

    # Transform → sparse matrix (1, n_features)
    X_vec = tfidf.transform([cleaned_text])

    # Get indices of non-zero features (words actually present in this text)
    nonzero_indices = X_vec.nonzero()[1]

    if len(nonzero_indices) == 0:
        return []

    scored = []
    for idx in nonzero_indices:
        tfidf_weight  = X_vec[0, idx]           # how prominent this word is in the text
        model_coef    = coefs[idx]               # how much the model weights this word
        contribution  = float(tfidf_weight * model_coef)  # actual push toward stress/calm
        word          = feature_names[idx]
        scored.append((word, contribution))

    # Sort by absolute contribution — biggest movers first
    scored.sort(key=lambda x: abs(x[1]), reverse=True)

    return [
        {
            'word'       : word,
            'impact'     : round(contribution, 4),
            'direction'  : 'increases stress' if contribution > 0 else 'reduces stress',
        }
        for word, contribution in scored[:n]
    ]


def _build_explanation(label: str, confidence: float, top_words: list[dict],
                        prob_low: float, prob_high: float) -> dict:
    """
    Builds a structured, human-readable explanation with multiple parts:
      - what_it_means : plain-language description of the confidence level
      - word_reason   : which words drove the result
      - confidence_breakdown : how the 0–100% score is composed
    """
    stress_words    = [w['word'] for w in top_words if w['direction'] == 'increases stress']
    nonstress_words = [w['word'] for w in top_words if w['direction'] == 'reduces stress']

    # ── Confidence tier description ───────────────────────────────────────
    if confidence >= 90:
        conf_desc = "very strong"
    elif confidence >= 75:
        conf_desc = "strong"
    elif confidence >= 60:
        conf_desc = "moderate"
    else:
        conf_desc = "mild"

    # ── Word-based reason sentence ────────────────────────────────────────
    if label == 'HIGH STRESS':
        if stress_words:
            word_reason = (
                f"The model detected {conf_desc} stress signals. "
                f"Key words like \"{', '.join(stress_words[:3])}\" carried heavy weight "
                f"toward a HIGH STRESS prediction."
            )
            if nonstress_words:
                word_reason += (
                    f" Some calming language was also present "
                    f"(\"{', '.join(nonstress_words[:2])}\"), "
                    f"but was outweighed by the stress indicators."
                )
        else:
            word_reason = (
                f"The model detected a {conf_desc} overall stress pattern "
                f"across your text, even without single standout words."
            )
    else:  # LOW STRESS
        if nonstress_words:
            word_reason = (
                f"The model found {conf_desc} signs of a calm or low-stress state. "
                f"Words like \"{', '.join(nonstress_words[:3])}\" contributed positively."
            )
            if stress_words:
                word_reason += (
                    f" Some stress-associated words appeared "
                    f"(\"{', '.join(stress_words[:2])}\"), "
                    f"but the overall tone leaned calm."
                )
        elif stress_words:
            word_reason = (
                f"Some stress-related words were detected "
                f"(\"{', '.join(stress_words[:2])}\"), "
                f"but not enough to cross the threshold for HIGH STRESS."
            )
        else:
            word_reason = (
                f"Your text showed a {conf_desc} low-stress pattern overall."
            )

    # ── Confidence breakdown explanation ─────────────────────────────────
    conf_breakdown = (
        f"The model assigned {prob_high:.1f}% probability to HIGH STRESS "
        f"and {prob_low:.1f}% to LOW STRESS. "
        f"It chose {label} as the final prediction because that had the higher score."
    )

    return {
        'word_reason'          : word_reason,
        'confidence_breakdown' : conf_breakdown,
        'confidence_tier'      : conf_desc,
        'stress_words'         : stress_words,
        'nonstress_words'      : nonstress_words,
    }


def explain_prediction(pipeline, raw_text: str, n: int = 5) -> dict:
    """
    Full prediction + rich explanation for a single input text.

    Returns:
        label                : 'HIGH STRESS' or 'LOW STRESS'
        confidence           : float (probability of predicted class, 0–100)
        prob_high            : float (probability of HIGH STRESS, 0–100)
        prob_low             : float (probability of LOW STRESS, 0–100)
        top_words            : list of {word, impact, direction}
        explanation          : dict with word_reason, confidence_breakdown, etc.
    """
    cleaned = clean_reddit_text(raw_text)

    pred     = pipeline.predict([cleaned])[0]
    probs    = pipeline.predict_proba([cleaned])[0]

    # predict_proba returns [prob_class0, prob_class1]
    prob_low  = round(float(probs[0]) * 100, 1)
    prob_high = round(float(probs[1]) * 100, 1)
    conf      = round(float(probs[pred]) * 100, 1)
    label     = 'HIGH STRESS' if pred == 1 else 'LOW STRESS'

    # Get top contributing words
    top_words = get_top_words(pipeline, cleaned, n)

    # Build rich explanation
    explanation = _build_explanation(label, conf, top_words, prob_low, prob_high)

    return {
        'label'      : label,
        'confidence' : conf,
        'prob_high'  : prob_high,
        'prob_low'   : prob_low,
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