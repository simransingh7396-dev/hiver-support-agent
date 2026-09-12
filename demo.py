"""Demo web app for AskPlayStation agent
Run: pip install streamlit
     streamlit run demo.py
"""
import streamlit as st
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from agent import Agent

st.set_page_config(page_title="AskPlayStation Support Agent", page_icon="🎮", layout="centered")
st.title("🎮 AskPlayStation Support Agent")
st.caption("Classify → Retrieve → Draft → Escalate | Hiver Intern Demo | Uses `data/resolved_memory.jsonl` (6,447 fixes)")

# Init agent once
@st.cache_resource
def get_agent():
    # use keyword fallback by default to avoid quota issues in demo; toggle below
    return Agent(use_llm=False)

agent = get_agent()

examples = [
    "@AskPlayStation Game keeps freezing and crashing error CE-34878-0",
    "@AskPlayStation I was banned for no reason please help",
    "@AskPlayStation I bought FIFA points but code says invalid when redeem",
    "@AskPlayStation Can't connect to server NAT type 3 on campus wifi",
    "@AskPlayStation Why was I charged £49.99 twice? Need refund",
]

with st.sidebar:
    st.header("Settings")
    use_llm = st.checkbox("Use Gemini (needs .env GEMINI_API_KEY, 20/day)", value=False)
    st.write("**Intents (7):** Technical, Account, Billing, Purchase, Network, Hardware, General")
    st.write("**Headline:** Intent 0.95, Escalation F1 0.943")
    st.write("**Repo:** [GitHub](https://github.com/simransingh7396-dev/hiver-support-agent)")
    if st.button("Clear cache"):
        st.cache_resource.clear()
        st.rerun()

# Update agent if toggle changed (simple: re-init)
if use_llm:
    try:
        agent_llm = Agent(use_llm=True)
        agent = agent_llm
        st.sidebar.success("Gemini on")
    except Exception as e:
        st.sidebar.warning(f"Gemini failed, using keyword: {e}")

query = st.text_area("Customer message (tweet):", value=examples[0], height=100, placeholder="Type @AskPlayStation ...")

col1, col2 = st.columns([1,3])
with col1:
    run = st.button("Run Agent 🚀", type="primary")
with col2:
    ex = st.selectbox("Try example:", examples, index=0)
    if st.button("Load example"):
        query = ex

if run and query.strip():
    with st.spinner("Running classify → retrieve → draft → escalate..."):
        res = agent.handle(query)
    # Pretty
    colA, colB = st.columns(2)
    with colA:
        st.metric("Intent", res["intent_name"])
        st.metric("Confidence", f'{res["confidence"]:.2f}')
    with colB:
        esc = "🔴 Escalate to human" if res["escalate"] else "🟢 Auto-handle"
        st.metric("Decision", esc)
        st.write(f"**Reason:** {res['escalate_reason']}")
    st.subheader("Draft Reply (grounded)")
    st.success(res["draft"])
    st.subheader("Retrieved 3 historical fixes")
    for i,r in enumerate(res["retrieved"],1):
        with st.expander(f"#{i} score {r['score']:.3f} thread {r['thread_id']}"):
            st.write(r["brand_resolution"])
    st.json(res)

st.divider()
st.caption("Tip for interview: try a paraphrase like 'my life in this account someone changed email' → should be Account Access & Security (escalate)")
