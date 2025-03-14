import streamlit as st
import os
from dotenv import load_dotenv
from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain_core.messages import HumanMessage, AIMessage, SystemMessage

load_dotenv()

st.set_page_config(page_title="Predictive Maintenance Chatbot")

st.title("🤖 Predictive Maintenance Chatbot")

# Initialize Pinecone
pc = Pinecone(api_key=os.environ.get("PINECONE_API_KEY"))
index_name = os.environ.get("PINECONE_INDEX_NAME")
index = pc.Index(index_name)

# Initialize embeddings model and vector store
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ.get("OPENAI_API_KEY")
)
vector_store = PineconeVectorStore(index=index, embedding=embeddings)

# Initialize LLM
llm = ChatOpenAI(model="gpt-4o", temperature=0.7)

# Initialize messages session state
if "messages" not in st.session_state:
    st.session_state.messages = [
        SystemMessage("You are an assistant specialized in predictive maintenance question-answering tasks.")
    ]


########
# # Check if there's a predicted_failure passed from the previous page
# if "predicted_failure" in st.session_state:
#     initial_query = f"What actions should I take if my machine has a '{st.session_state['predicted_failure']}'?"
#     st.session_state.messages.append(HumanMessage(initial_query))
#     del st.session_state["predicted_failure"]
# if "predicted_failure" in st.session_state and not any(isinstance(msg, HumanMessage) and st.session_state['predicted_failure'] in msg.content for msg in st.session_state.messages):
#     initial_query = f"What actions should I take if my machine has a '{st.session_state['predicted_failure']}'?"
#     st.session_state.messages.append(HumanMessage(initial_query))
#     del st.session_state["predicted_failure"]


# Display previous chat messages
for message in st.session_state.messages:
    if isinstance(message, HumanMessage):
        with st.chat_message("user"):
            st.markdown(message.content)
    elif isinstance(message, AIMessage):
        with st.chat_message("assistant"):
            st.markdown(message.content)

# User input prompt
prompt = st.chat_input("How can I help you?")

if prompt:
    with st.chat_message("user"):
        st.markdown(prompt)
        st.session_state.messages.append(HumanMessage(prompt))

    # Initialize LLM
    llm = ChatOpenAI(model="gpt-4o", temperature=1)

    # Retrieve relevant documents based on user prompt
    retriever = vector_store.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": 5, "score_threshold": 0.5}
    )

    docs = retriever.invoke(prompt)
    docs_text = "".join(doc.page_content for doc in docs)

    system_prompt = """
    You are an assistant specialized in predictive maintenance.
    Use the provided context to answer the question.
    If the context does not contain the answer, say that you don't know.
    Limit your response to a maximum of ten concise sentences.

    Context: {context}
    """

    # format the system prompt with the retrieved context
    system_prompt_fmt = system_prompt.format(context=docs_text)

    # Append system prompt to the conversation history
    st.session_state.messages.append(SystemMessage(system_prompt_fmt))

    # Get response from the LLM
    response = llm.invoke(st.session_state.messages).content

    with st.chat_message("assistant"):
        st.markdown(response)
        st.session_state.messages.append(AIMessage(response))
