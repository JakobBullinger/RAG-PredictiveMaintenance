import os, time
from dotenv import load_dotenv

from pinecone import Pinecone
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings, ChatOpenAI
from langchain.chains import ConversationalRetrievalChain

load_dotenv()  # loads keys from .env

# ── 1. Vector store ─────────────────────────────────────────────────────────
pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
index = pc.Index(os.environ["PINECONE_INDEX_NAME"])
embeddings = OpenAIEmbeddings(
    model="text-embedding-3-large",
    api_key=os.environ["OPENAI_API_KEY"]
)
VECTOR_STORE = PineconeVectorStore(index=index, embedding=embeddings)

# ── 2. LLM ------------------------------------------------------------------
LLM = ChatOpenAI(model="gpt-3.5-turbo", temperature=0.7)

def rag_answer(question: str, chat_history=None, k: int = 3,
               score_threshold: float = 0.5):
    """
    Run one RAG cycle and return (answer, sources, latency_in_seconds).
    """
    chat_history = chat_history or []
    retriever = VECTOR_STORE.as_retriever(
        search_type="similarity_score_threshold",
        search_kwargs={"k": k, "score_threshold": score_threshold}
    )
    chain = ConversationalRetrievalChain.from_llm(
        LLM, retriever, return_source_documents=True
    )

    t0 = time.perf_counter()
    result = chain({"question": question, "chat_history": chat_history})
    dt = time.perf_counter() - t0

    sources = [
        doc.metadata.get("source", "unknown") for doc in result["source_documents"]
    ]
    return result["answer"], sources, dt
