import streamlit as st
import torch
import torch.nn.functional as F
from transformers import GPT2Tokenizer, GPT2LMHeadModel

# ---------------------------------------------------------
# FRAMEHOUSE PRESS — Literary Submission Portal
# ---------------------------------------------------------

# ---- Load fine-tuned model ----
MODEL_PATH = "./gpt2_literary_dual"
tokenizer = GPT2Tokenizer.from_pretrained(MODEL_PATH)
tokenizer.pad_token = tokenizer.eos_token
model = GPT2LMHeadModel.from_pretrained(MODEL_PATH)
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model.to(device)
model.eval()

# ---- Core Functions ----
def score_text(text, prefix):
    """Compute average log-probability of text given a prefix."""
    input_text = f"{prefix} {text}"
    inputs = tokenizer(input_text, return_tensors="pt").to(device)
    with torch.no_grad():
        outputs = model(**inputs)
        logits = outputs.logits[:, :-1, :]
        labels = inputs.input_ids[:, 1:]
        log_probs = F.log_softmax(logits, dim=-1)
        selected_log_probs = log_probs.gather(2, labels.unsqueeze(-1)).squeeze(-1)
        total_log_prob = selected_log_probs.sum().item()
        avg_log_prob = total_log_prob / labels.size(1)
    return avg_log_prob

def classify_passage(passage):
    """Compare how 'ideal' vs 'nonfit' the writing is."""
    score_ideal = score_text(passage, "[ideal]")
    score_nonfit = score_text(passage, "[nonfit]")
    label = "ideal" if score_ideal > score_nonfit else "nonfit"
    return label, score_ideal, score_nonfit

def generate_continuation(passage):
    """Generate an ideal-style continuation."""
    full_prompt = f"[ideal] {passage}"
    inputs = tokenizer(full_prompt, return_tensors="pt").to(device)
    outputs = model.generate(
        inputs.input_ids,
        max_length=128,
        do_sample=True,
        top_k=50,
        top_p=0.95,
        num_return_sequences=1
    )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# ---------------------------------------------------------
# 🪶 STREAMLIT INTERFACE DESIGN
# ---------------------------------------------------------

st.set_page_config(
    page_title="Framehouse Press — Submission Portal",
    layout="centered",
    page_icon="🪶",
)

# ---- Custom CSS for cinematic atmosphere ----
st.markdown("""
    <style>
    body {
        background-color: #f8f5f2;
        font-family: 'Georgia', serif;
        color: #2b2b2b;
    }
    .title {
        font-size: 40px;
        font-weight: bold;
        text-align: center;
        color: #2b2b2b;
        padding-bottom: 0px;
        margin-bottom: 0;
    }
    .subtitle {
        text-align: center;
        font-size: 18px;
        color: #666666;
        margin-bottom: 40px;
    }
    .highlight {
        color: #8b5e3c;
        font-weight: bold;
    }
    .result-box {
        border: 1px solid #ddd;
        padding: 10px;
        background-color: #faf8f5;
        border-radius: 10px;
    }
    </style>
""", unsafe_allow_html=True)

# ---- Header ----
st.markdown("<div class='title'>Framehouse Press</div>", unsafe_allow_html=True)
st.markdown("<div class='subtitle'>A Cinematic Literary Press — Testing Voice and Vision</div>", unsafe_allow_html=True)
st.write("")

# ---- Input area ----
st.markdown("### 📜 Writer’s Submission")
text = st.text_area(" ", placeholder="Paste your prose or poem here...", height=220)

col1, col2 = st.columns(2)
with col1:
    analyze = st.button("🔍 Analyze Submission")
with col2:
    clear = st.button("🧹 Clear")

if clear:
    st.experimental_rerun()

# ---- Output ----
if analyze and text.strip():
    with st.spinner("Analyzing tone and style..."):
        continuation = generate_continuation(text)
        label, s_i, s_n = classify_passage(text)

    st.markdown("### 🪞 Model Continuation (Ideal Style)")
    st.text_area(" ", continuation, height=150)

    st.markdown("### 🧭 Classification Results")
    color = "green" if label == "ideal" else "red"
    st.markdown(f"**Predicted Label:** <span style='color:{color}; font-weight:bold;'>{label.upper()}</span>", unsafe_allow_html=True)
    st.markdown(f"**Score [ideal]:** {s_i:.2f}")
    st.markdown(f"**Score [nonfit]:** {s_n:.2f}")

    st.markdown("---")
    st.markdown("### 🖋️ Editor Notes")
    notes = st.text_area("Leave optional editorial notes...", height=100)
    if st.button("💾 Save to Archive"):
        st.success("Submission and notes saved (local archive feature can be enabled).")

elif analyze and not text.strip():
    st.warning("Please paste some text before analyzing.")

# ---- Footer ----
st.markdown("---")
st.markdown("<div style='text-align:center; color:#888888; font-size:14px;'>© Framehouse Press & Design — Literary AI Studio</div>", unsafe_allow_html=True)
