import os
import streamlit as st
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage
from langchain.chains import ConversationalRetrievalChain

load_dotenv()

st.set_page_config(
    page_title="Predictive Maintenance Chatbot")

st.title("🤖 Predictive Maintenance Chatbot")


pc = Pinecone(
    api_key     = os.environ["PINECONE_API_KEY"],
    environment = os.environ["PINECONE_ENVIRONMENT"], 
)
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])

# 2) Wrap in LangChain
embeddings = OpenAIEmbeddings(
    model   = "text-embedding-3-large",
    api_key = os.environ["OPENAI_API_KEY"],
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)


# 2) Initialize the LLM ─
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# ── 3) Chat history in session_state 
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage("You are an assistant specialized in predictive maintenance question-answering tasks.")
    ]

# 3a) Inject any incoming query param
params = st.query_params
if "q" in params and params["q"]:
    initial = params["q"][0]
    # only add it once
    if not any(isinstance(m, HumanMessage) and m.content == initial 
               for m in st.session_state.messages):
        st.session_state.messages.append(HumanMessage(initial))
    # clear params so we don’t re-inject on reload
    st.query_params = {}

# 3b) Display any existing messages
for msg in st.session_state.messages:
    if isinstance(msg, HumanMessage):
        with st.chat_message("user"):
            st.markdown(msg.content)
    elif isinstance(msg, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(msg.content)

# 4) Handle new user input 
prompt = st.chat_input("How can I help you?")

if prompt:
    # Append user message to history & UI
    st.session_state.messages.append(HumanMessage(prompt))
    with st.chat_message("user"):
        st.markdown(prompt)

    # Build a retriever and a conversational RAG chain
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )
    chain = ConversationalRetrievalChain.from_llm(
        llm, retriever, return_source_documents=True
    )

    # Convert session_state.messages
    history = []
    qa = {}
    for m in st.session_state.messages:
        if isinstance(m, HumanMessage):
            qa["q"] = m.content
        elif isinstance(m, AIMessage) and qa.get("q"):
            qa["a"] = m.content
            history.append((qa["q"], qa["a"]))
            qa = {}

    # Invoke the chain
    try:
        result = chain({"question": prompt, "chat_history": history})
        answer = result["answer"]
        sources = result["source_documents"]
    except Exception as e:
        st.error(f"Something went wrong: {e}")
        answer = None
        sources = []

    # Display the answer
    if answer:
        st.session_state.messages.append(AIMessage(answer))
        with st.chat_message("assistant"):
            st.markdown(answer)

 
        if sources:
            st.markdown("**Sources:**")
            for i, doc in enumerate(sources, start=1):
                src = doc.metadata.get("source", "unknown")
                pg = doc.metadata.get("page")
                loc = f" (page {pg})" if pg else ""
                st.markdown(f"{i}. {src}{loc}")
