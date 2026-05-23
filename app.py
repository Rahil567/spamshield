# SpamShield: A Streamlit app for real-time spam detection using a logistic regression model trained on TF-IDF features.

import streamlit as st
import pandas as pd

# PAGE CONFIG 

st.set_page_config(page_title="SpamShield", page_icon="🔥", layout="centered")

# STYLES 

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Bebas+Neue&family=DM+Mono:wght@400;500&family=DM+Sans:wght@400;500;700&display=swap');

html, body, [class*="css"]          { font-family: 'DM Sans', sans-serif; }
.stApp                              { background: #080808; color: #e8e0d8; }
#MainMenu, footer, header           { visibility: hidden; }

/* ── hero ── */
.hero { text-align:center; padding: 2.5rem 0 1.5rem; }
.hero-logo {
    font-family:'Bebas Neue',sans-serif; font-size:4rem; letter-spacing:4px;
    background: linear-gradient(135deg,#ff4500 0%,#ff8c00 60%,#ffd700 100%);
    -webkit-background-clip:text; -webkit-text-fill-color:transparent;
    display:inline-block; line-height:1;
}
.hero-tag {
    font-family:'DM Mono',monospace; font-size:.72rem; letter-spacing:3px;
    color:#555; margin-top:.25rem;
}

/* ── textarea ── */
textarea {
    background:#111 !important; color:#e8e0d8 !important;
    border:1px solid #2a2a2a !important; border-radius:10px !important;
    font-family:'DM Mono',monospace !important; font-size:.9rem !important;
}
textarea:focus { border-color:#ff4500 !important; box-shadow:0 0 0 2px rgba(255,69,0,.25) !important; }

/* ── button ── */
.stButton>button {
    width:100%; background:linear-gradient(135deg,#ff4500,#ff8c00);
    color:#fff; font-family:'DM Sans',sans-serif; font-weight:700;
    font-size:1rem; letter-spacing:1px; border:none; border-radius:10px;
    padding:.65rem 3rem; transition:opacity .2s, transform .1s;
}
.stButton>button:hover  { opacity:.88; transform:translateY(-1px); }
.stButton>button:active { transform:translateY(0); }

/* ── result card ── */
.result-card {
    border-radius:14px; padding:1.5rem 1.75rem; margin-top:1.2rem;
    border-left:5px solid;
}
.result-spam { background:#1a0a00; border-color:#ff4500; }
.result-ham  { background:#050f07; border-color:#22c55e; }
.result-label { font-family:'Bebas Neue',sans-serif; font-size:2rem; letter-spacing:2px; }
.result-spam .result-label { color:#ff4500; }
.result-ham  .result-label { color:#22c55e; }
.result-prob { font-family:'DM Mono',monospace; font-size:.85rem; color:#888; margin-top:.25rem; }

/* ── metric tiles ── */
.metric-row { display:flex; gap:.75rem; margin:1.2rem 0; }
.metric-tile {
    flex:1; background:#111; border:1px solid #1e1e1e; border-radius:12px;
    padding:.9rem .5rem; text-align:center;
}
.metric-tile .val { font-family:'Bebas Neue',sans-serif; font-size:1.9rem; letter-spacing:1px; color:#ff6a00; }
.metric-tile .lbl { font-size:.7rem; color:#555; letter-spacing:2px; text-transform:uppercase; margin-top:.1rem; }

/* ── feature bar ── */
.feat-row { display:flex; align-items:center; gap:.75rem; margin:.4rem 0; }
.feat-word { font-family:'DM Mono',monospace; font-size:.8rem; color:#ccc; width:160px; flex-shrink:0; }
.feat-bar-bg { flex:1; background:#1a1a1a; border-radius:999px; height:8px; overflow:hidden; }
.feat-bar-fill { height:100%; border-radius:999px; background:linear-gradient(90deg,#ff4500,#ff8c00); }
.feat-score { font-family:'DM Mono',monospace; font-size:.75rem; color:#555; width:42px; text-align:right; }

/* ── tabs ── */
.stTabs [data-baseweb="tab-list"] { background:#0e0e0e; border-radius:10px; padding:.5rem; display:flex; }
.stTabs [data-baseweb="tab"]      { flex:1; text-align:center; border-radius:8px; color:#666; font-size:.85rem; }
.stTabs [aria-selected="true"]    { background:#1a1a1a !important; color:#ff6a00 !important; }

/* ── expander ── */
details                            { background:#111 !important; border-radius:10px !important; border:1px solid #1e1e1e !important; }
details summary                    { background:#111 !important; border-radius:10px !important; padding:.75rem 1rem !important; }
details summary p                  { color:#ff6a00 !important; font-weight:600 !important; }
details[open] summary              { border-radius:10px 10px 0 0 !important; }
details > div                      { background:#111 !important; border-radius:0 0 10px 10px !important; }
</style>
""", unsafe_allow_html=True)

# LOAD MODEL 

@st.cache_resource(show_spinner=False)
def load():
    import E_spam
    E_spam._load()
    return E_spam

with st.spinner("Warming up the model…"):
    spam = load()

# HERO 

st.markdown("""
<div class="hero">
  <div class="hero-logo">🔥 SpamShield</div>
  <div class="hero-tag">TF-IDF · LOGISTIC REGRESSION · REAL-TIME DETECTION</div>
</div>
""", unsafe_allow_html=True)

# TABS

tab_detect, tab_bulk, tab_insights = st.tabs(["Detect", "Bulk Scan", "Model Insights"])

#  TAB 1 — DETECT

with tab_detect:
    st.markdown("<br>", unsafe_allow_html=True)
    text = st.text_area("Message", height=160,
                        placeholder="Paste any SMS, email, or message here…",
                        label_visibility="collapsed")

    if st.button("Try an example ↗", key="ex"):
        st.session_state["example_text"] = "WINNER!! As a valued customer you've been selected to receive a £900 prize! Call 09061701461 NOW!"

    if "example_text" in st.session_state:
        text = st.session_state.pop("example_text")
        st.rerun()

    col_btn, _ = st.columns([1, 2])
    with col_btn:
        analyse = st.button("Analyse ⚡")

    if analyse:
        if not text.strip():
            st.warning("Please enter a message first.")
        else:
            pred, prob = spam.predict_spam(text)
            is_spam  = pred == 1
            pct      = int(prob * 100)
            card_cls = "result-spam" if is_spam else "result-ham"
            label    = "🚨 SPAM" if is_spam else "✅ NOT SPAM"
            note     = "High likelihood of unsolicited/malicious content." if is_spam \
                       else "Looks legitimate — no spam signals detected."

            st.markdown(f"""
            <div class="result-card {card_cls}">
              <div class="result-label">{label}</div>
              <div class="result-prob">Spam probability: {pct}%</div>
              <div style="margin-top:.6rem;font-size:.85rem;color:#aaa">{note}</div>
            </div>
            """, unsafe_allow_html=True)
            st.progress(prob)

#  TAB 2 — BULK SCAN

with tab_bulk:
    st.markdown("<br>", unsafe_allow_html=True)
    st.caption("One message per line. Paste up to 50 messages.")
    bulk = st.text_area("Messages", height=220,
                        placeholder="Message 1\nMessage 2\nMessage 3…",
                        label_visibility="collapsed")

    if st.button("Scan All 🚀", key="bulk_btn"):
        lines = [l.strip() for l in bulk.splitlines() if l.strip()][:50]
        if not lines:
            st.warning("Enter at least one message.")
        else:
            rows, prog = [], st.progress(0)
            for i, line in enumerate(lines):
                p, prob = spam.predict_spam(line)
                rows.append({"Message": line[:80] + ("…" if len(line) > 80 else ""),
                             "Result":  "🚨 Spam" if p else "✅ Ham",
                             "Spam %":  f"{prob*100:.1f}%"})
                prog.progress((i + 1) / len(lines))
            prog.empty()

            st.dataframe(pd.DataFrame(rows), use_container_width=True)

            spam_n = sum(1 for r in rows if "🚨" in r["Result"])
            st.markdown(f"""
            <div class="metric-row">
              <div class="metric-tile"><div class="val">{len(rows)}</div><div class="lbl">Total</div></div>
              <div class="metric-tile"><div class="val" style="color:#ff4500">{spam_n}</div><div class="lbl">Spam</div></div>
              <div class="metric-tile"><div class="val" style="color:#22c55e">{len(rows)-spam_n}</div><div class="lbl">Ham</div></div>
            </div>""", unsafe_allow_html=True)

#  TAB 3 — MODEL INSIGHTS

with tab_insights:
    st.markdown("<br>", unsafe_allow_html=True)

    if "metrics" not in st.session_state:
        with st.spinner("Computing metrics…"):
            st.session_state.metrics = spam.get_model_metrics()
    m = st.session_state.metrics

    st.markdown(f"""
    <div class="metric-row">
      <div class="metric-tile"><div class="val">{m['accuracy']}%</div><div class="lbl">Accuracy</div></div>
      <div class="metric-tile"><div class="val">{m['precision']}%</div><div class="lbl">Precision</div></div>
      <div class="metric-tile"><div class="val">{m['recall']}%</div><div class="lbl">Recall</div></div>
      <div class="metric-tile"><div class="val">{m['f1']}%</div><div class="lbl">F1 Score</div></div>
    </div>""", unsafe_allow_html=True)

    st.markdown("#### Top Spam Indicator Words")
    features = spam.top_spam_features(12)
    max_coef = features[0][1] if features else 1
    bars_html = ""
    for word, score in features:
        pct = int((score / max_coef) * 100)
        bars_html += f"""
        <div class="feat-row">
          <div class="feat-word">{word}</div>
          <div class="feat-bar-bg"><div class="feat-bar-fill" style="width:{pct}%"></div></div>
          <div class="feat-score">{score}</div>
        </div>"""
    st.markdown(bars_html, unsafe_allow_html=True)

    with st.expander("Classification Report"):
        report_html = m["report"].replace(" ", "&nbsp;").replace("\n", "<br>")
        st.markdown(
            f'<div style="background:#000;color:#e8e0d8;padding:1.2rem 1.5rem;border-radius:8px;'
            f'font-family:DM Mono,monospace;font-size:.82rem;line-height:1.8;">'
            f'{report_html}</div>',
            unsafe_allow_html=True)