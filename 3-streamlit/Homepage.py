import pathlib
import streamlit as st

# ── Page config & styling ────────────────────────────────────────────────────
st.set_page_config(page_title="Predictive Maintenance Dashboard", layout="wide")
st.markdown(
    """
    <style>
      /* Center the header text */
      h1 { text-align: center; }
      /* Full-width buttons in columns */
      .stButton>button { width: 100%; }
    </style>
    """,
    unsafe_allow_html=True,
)

# ── Paths ────────────────────────────────────────────────────────────────────
HERE = pathlib.Path(__file__).parent
IMG  = HERE / "assets" / "architecture.png"

# ── 1) Hero Section ─────────────────────────────────────────────────────────
st.markdown("# Predictive Maintenance Dashboard")
st.markdown(
    """
    <div style="display: flex; justify-content: center; margin: 1rem 0;">
      <a href="/Predictive_Maintenance_Algorithm" target="_self">
        <button style="
          background-color:#FFFFFF;
          color:black;
          padding: 10px 20px;
          border-radius:8px;
          border:none;
          cursor:pointer;
          font-size:1rem;
        ">
          ▶️ Launch Failure Classifier
        </button>
      </a>
    </div>
    """,
    unsafe_allow_html=True
)
st.write("---")

# ── 2) System Architecture Diagram ───────────────────────────────────────────
col1, col2 = st.columns([2, 3])
with col1:
    if IMG.exists():
        st.image(
            str(IMG),
            caption="Data → Model → RAG Index → Chatbot",
            use_container_width=True
        )
    else:
        st.warning(f"Add `{IMG}` to view the system diagram.")
with col2:
    st.markdown("### How It Works")
    st.markdown(
        """
        1. **Data Ingestion**  
           Load sensor logs, maintenance records & equipment manuals.  
        2. **Predictive Model**  
           A gradient-boosting classifier forecasts healthy/minor/critical faults.  
        3. **RAG Indexing**  
           Token-chunked manuals embedded with OpenAI and stored in Pinecone.  
        4. **AI Chatbot**  
           Ask questions and get context-aware maintenance recommendations.
        """
    )

st.write("---")

# ── 3) Thesis Objectives ─────────────────────────────────────────────────────
st.markdown("### Thesis Objectives")
obj_cols = st.columns(3)
titles = ["🔍 Integration", "🚧 Challenges", "💼 Implications"]
descs = [
    "Fuse predictive-maintenance algorithms with an RAG chatbot interface.",
    "Evaluate operational benefits & potential workflow hurdles.",
    "Analyze enterprise value, cost impacts & deployment scalability."
]
for col, title, desc in zip(obj_cols, titles, descs):
    with col:
        st.markdown(f"**{title}**")
        st.write(desc)


# ── 4) Footer ────────────────────────────────────────────────────────────────
st.markdown(
    """
    <hr style="margin: 1em 0;">
    Built by Jakob Bullinger · University of St. Gallen  
    Powered by Streamlit • Pinecone • OpenAI • LangChain
    """,
    unsafe_allow_html=True
)
