import os
import time
from dotenv import load_dotenv

from pinecone import Pinecone, ServerlessSpec
from langchain_pinecone import PineconeVectorStore
from langchain_openai import OpenAIEmbeddings
from langchain_community.document_loaders import PyPDFDirectoryLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

import tiktoken

def num_tokens(text: str) -> int:
    """Return number of tokens in a text using the CL100K tokenizer."""
    return len(tokenizer.encode(text))

if __name__ == "__main__":
    load_dotenv()

    # ── 1. Connect to Pinecone & ensure index exists ───────────────────────────
    try:
        pc = Pinecone(api_key=os.environ["PINECONE_API_KEY"])
        index_name = os.environ["PINECONE_INDEX_NAME"]
        existing = [idx["name"] for idx in pc.list_indexes()]

        if index_name not in existing:
            print(f"Creating Pinecone index '{index_name}'…")
            pc.create_index(
                name=index_name,
                dimension=3072,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region="us-east-1"),
            )
            # wait until ready
            while not pc.describe_index(index_name).status["ready"]:
                time.sleep(1)

        index = pc.Index(index_name)
        print(f"Connected to Pinecone index '{index_name}'.")
    except Exception as e:
        print(f"Failed to initialize Pinecone: {e}")
        raise SystemExit(e)

    # ── 2. Set up embeddings + vector store ────────────────────────────────────
    embeddings = OpenAIEmbeddings(
        model="text-embedding-3-large",
        api_key=os.environ["OPENAI_API_KEY"]
    )
    vector_store = PineconeVectorStore(index=index, embedding=embeddings)

    # ── 3. Load your PDFs ──────────────────────────────────────────────────────
    loader = PyPDFDirectoryLoader("2-rag/documents/")
    raw_documents = loader.load()
    print(f"Loaded {len(raw_documents)} raw document(s).")

    # ── 4. Token-based splitting ───────────────────────────────────────────────
    tokenizer = tiktoken.get_encoding("cl100k_base")
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,            # 500 tokens per chunk
        chunk_overlap=100,         # 100-token overlap
        length_function=num_tokens,
        is_separator_regex=False,
    )
    documents = text_splitter.split_documents(raw_documents)
    print(f"Split into {len(documents)} chunks (≈500 tokens each).")

    # ── 5. Generate IDs & add to Pinecone ─────────────────────────────────────
    ids = [f"id{i+1}" for i in range(len(documents))]
    try:
        vector_store.add_documents(documents=documents, ids=ids)
        print(f"Upserted {len(documents)} embeddings into Pinecone.")
    except Exception as e:
        print(f"Error upserting documents: {e}")
        raise
