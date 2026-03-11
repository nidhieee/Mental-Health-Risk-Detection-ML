"""
Stress Detection App — Streamlit UI
Run: streamlit run app.py
"""

import streamlit as st
import pickle
import os
import sys

# ─── Import from your pipeline ───────────────────────────────────────────────
# Place this file in the same folder as stress_model_pipeline.py
sys.path.insert(0, os.path.dirname(__file__))
from stress_model_pipeline import clean_reddit_text, explain_prediction, load_model


# ════════════════════════════════════════════════════════════════════════════
# Recommendations (rule-based, maps to keywords)
# ════════════════════════════════════════════════════════════════════════════
RECOMMENDATIONS = {
    # Sleep-related
    'sleep': ("😴 Better Sleep", "Try a fixed sleep schedule — same time every night."),
    'tired': ("😴 Rest", "Take a 20-min power nap. Even short rest recharges you."),
    'exhausted': ("😴 Rest", "Prioritize sleep tonight. Turn screens off 30 mins before bed."),
    'insomnia': ("😴 Sleep Hygiene", "Try 4-7-8 breathing: inhale 4s, hold 7s, exhale 8s."),

    # Anxiety / overwhelm
    'overwhelm': ("🧘 Calm Down", "Try box breathing: 4s in, 4s hold, 4s out, 4s hold."),
    'anxious': ("🧘 Grounding", "5-4-3-2-1 technique: name 5 things you see, 4 you hear..."),
    'panic': ("🧘 Breathe", "Breathe in slowly for 4 counts, out for 6. Repeat 5 times."),
    'worry': ("📓 Journaling", "Write your worries down. Getting them out of your head helps."),

    # Hopelessness
    'hopeless': ("💬 Talk to Someone", "Reaching out to a friend or counsellor can truly help."),
    'worthless': ("💬 Support", "You matter. Consider speaking with someone you trust today."),
    'alone': ("🤝 Connection", "Reach out to one person today — a text, a call, anything."),

    # General stress
    'stress': ("🧘 Yoga", "Even 10 minutes of yoga can lower cortisol levels noticeably."),
    'pressure': ("🎵 Music", "Listen to calming music. It lowers heart rate within minutes."),
    'deadline': ("📋 Plan It", "Write down tasks and pick just ONE to start. Small steps work."),

    # Body / physical
    'headache': ("💧 Hydrate", "Drink a full glass of water. Dehydration often causes headaches."),
    'pain': ("🚶 Walk", "A 10-minute walk outside reduces physical tension significantly."),
}

DEFAULT_HIGH_SUGGESTIONS = [
    ("🧘 Breathing Exercise", "Try box breathing: 4s in → 4s hold → 4s out → 4s hold. Repeat 4 times."),
    ("🚶 Take a Walk", "Step outside for 10 minutes. Fresh air and movement work fast."),
    ("💬 Talk to Someone", "A friend, family member, or counsellor can make a big difference."),
]

DEFAULT_LOW_SUGGESTIONS = [
    ("🌟 Keep It Up", "You seem to be managing well. Keep your healthy routines going!"),
    ("📓 Gratitude Journal", "Jot down 3 things you're grateful for today to reinforce positivity."),
    ("🎵 Enjoy the Moment", "Do something you love today — you deserve it."),
]

HELPLINE = "☎️  iCall (India): 9152987821  |  Vandrevala Foundation: 1860-2662-345"


def get_suggestions(label: str, top_words: list) -> list[tuple]:
    """Match top stress-inducing words to relevant suggestions."""
    suggestions = []
    stress_words = [w['word'] for w in top_words if w['direction'] == 'increases stress']

    for word in stress_words:
        for key, suggestion in RECOMMENDATIONS.items():
            if key in word and suggestion not in suggestions:
                suggestions.append(suggestion)
                break

    # Fill up to 3 suggestions with defaults
    defaults = DEFAULT_HIGH_SUGGESTIONS if label == 'HIGH STRESS' else DEFAULT_LOW_SUGGESTIONS
    for s in defaults:
        if len(suggestions) >= 3:
            break
        if s not in suggestions:
            suggestions.append(s)

    return suggestions[:3]


# ════════════════════════════════════════════════════════════════════════════
# Page Config
# ════════════════════════════════════════════════════════════════════════════
st.set_page_config(
    page_title="Stress Detector",
    page_icon="🧠",
    layout="centered"
)

st.markdown("""
<style>
    .big-result { font-size: 2rem; font-weight: 700; text-align: center; padding: 1rem; border-radius: 12px; }
    .high { background-color: #ffe4e4; color: #c0392b; }
    .low  { background-color: #e4f9e4; color: #1e8449; }
    .word-pill {
        display: inline-block; padding: 4px 12px; margin: 4px;
        border-radius: 20px; font-size: 0.85rem; font-weight: 600;
    }
    .stress-word    { background: #ffe4e4; color: #c0392b; }
    .nonstress-word { background: #e4f9e4; color: #1e8449; }
    .suggestion-card {
        background: #f8f9fa; border-left: 4px solid #6c63ff;
        padding: 0.7rem 1rem; margin: 0.4rem 0; border-radius: 8px;
    }
    .disclaimer {
        font-size: 0.78rem; color: #888; text-align: center;
        margin-top: 2rem; border-top: 1px solid #eee; padding-top: 1rem;
    }
</style>
""", unsafe_allow_html=True)


# ════════════════════════════════════════════════════════════════════════════
# Load Model
# ════════════════════════════════════════════════════════════════════════════
@st.cache_resource
def get_model():
    model_path = 'stress_model.pkl'
    if not os.path.exists(model_path):
        st.error("❌ Model file not found. Run `python stress_model_pipeline.py dataset.csv` first.")
        st.stop()
    return load_model(model_path)

pipeline = get_model()


# ════════════════════════════════════════════════════════════════════════════
# UI
# ════════════════════════════════════════════════════════════════════════════
st.title("Stress Level Detector")
st.caption("Type how you're feeling — we'll assess your stress level and suggest ways to help.")

user_text = st.text_area(
    "How are you feeling right now?",
    placeholder="e.g. I've been feeling overwhelmed with deadlines and can't sleep properly...",
    height=150
)

analyze_btn = st.button("Analyze", type="primary", use_container_width=True)

if analyze_btn:
    if not user_text.strip():
        st.warning("Please type something first.")
    elif len(user_text.split()) < 5:
        st.warning("Please write a bit more so the model can analyze accurately (at least 5 words).")
    else:
        with st.spinner("Analyzing..."):
            result = explain_prediction(pipeline, user_text)

        label      = result['label']
        confidence = result['confidence']
        top_words  = result['top_words']
        explanation = result['explanation']
        suggestions = get_suggestions(label, top_words)

        # ── Result Banner ─────────────────────────────────────────────────
        css_class = 'high' if label == 'HIGH STRESS' else 'low'
        emoji     = '🔴' if label == 'HIGH STRESS' else '🟢'
        st.markdown(f"""
            <div class="big-result {css_class}">
                {emoji} {label}
                <div style="font-size:1rem; font-weight:400; margin-top:4px;">
                    {confidence}% confidence
                </div>
            </div>
        """, unsafe_allow_html=True)

        st.markdown("---")

        # ── Explanation ───────────────────────────────────────────────────
        st.subheader("🔍 Why this result?")
        st.write(explanation)

        if top_words:
            st.markdown("**Key words detected:**")
            pills = ""
            for w in top_words:
                css = 'stress-word' if w['direction'] == 'increases stress' else 'nonstress-word'
                pills += f'<span class="word-pill {css}">{w["word"]}</span>'
            st.markdown(pills, unsafe_allow_html=True)

        st.markdown("---")

        # ── Suggestions ───────────────────────────────────────────────────
        st.subheader("💡 What you can do")
        for title, tip in suggestions:
            st.markdown(f"""
                <div class="suggestion-card">
                    <strong>{title}</strong><br>{tip}
                </div>
            """, unsafe_allow_html=True)

        # ── High stress helpline ──────────────────────────────────────────
        if label == 'HIGH STRESS':
            st.error(f"If you're in crisis, please reach out: {HELPLINE}")

        st.markdown("""
            <div class="disclaimer">
                ⚠️ This tool is for early awareness only — not a medical diagnosis.
                Always consult a qualified professional for mental health support.
            </div>
        """, unsafe_allow_html=True)