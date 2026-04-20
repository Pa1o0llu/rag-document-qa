import streamlit as st
import os
from dotenv import load_dotenv

from langchain_community.document_loaders import PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_groq import ChatGroq

# -------------------------------
# LOAD ENV VARIABLES
# -------------------------------
load_dotenv()
GROQ_API_KEY = os.getenv("GROQ_API_KEY")

# -------------------------------
# PAGE CONFIG
# -------------------------------
st.set_page_config(page_title="RAG Q&A", page_icon="📄", layout="wide")

st.title("📄 Smart Document Q&A System")
st.markdown("Upload a PDF and ask questions based on its content.")

# -------------------------------
# SIDEBAR
# -------------------------------
st.sidebar.header("📂 Upload Document")
uploaded_file = st.sidebar.file_uploader("Upload your PDF", type="pdf")

# -------------------------------
# LOAD LLM (FAST + GOOD QUALITY)
# -------------------------------
@st.cache_resource
def load_llm():
    return ChatGroq(
        api_key=GROQ_API_KEY,
        model_name="llama-3.1-8b-instant",
        temperature=0.1
    )

llm = load_llm()

# -------------------------------
# MAIN LOGIC
# -------------------------------
if uploaded_file is not None:

    # Save uploaded file
    with open("temp.pdf", "wb") as f:
        f.write(uploaded_file.read())

    st.success("✅ File uploaded successfully!")

    # Process document
    with st.spinner("Processing document..."):

        loader = PyPDFLoader("temp.pdf")
        documents = loader.load()

        text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=500,
            chunk_overlap=50
        )

        texts = text_splitter.split_documents(documents)

        embeddings = HuggingFaceEmbeddings(
            model_name="all-MiniLM-L6-v2"
        )

        db = Chroma.from_documents(texts, embeddings)

    st.info(f"📊 Document processed: {len(texts)} chunks created")

    # -------------------------------
    # USER QUERY
    # -------------------------------
    query = st.text_input("💬 Ask your question:")

    if query:
        with st.spinner("Generating answer..."):

            # Retrieve relevant chunks
            docs = db.similarity_search(query, k=4)

            context = "\n\n".join([doc.page_content for doc in docs])

            # Prompt for LLM
            prompt = f"""
You are a helpful assistant.

Answer the question using ONLY the context below.

If the answer is not present, say:
"I couldn't find this in the document."

Context:
{context}

Question:
{query}

Answer clearly in bullet points:
"""

            response = llm.invoke(prompt)

        # -------------------------------
        # OUTPUT
        # -------------------------------
        st.success("✅ Answer Generated")

        st.subheader("📌 Answer")
        st.markdown(response.content)

        # Optional: show retrieved chunks
        with st.expander("📚 Retrieved Context"):
            for i, doc in enumerate(docs, 1):
                st.markdown(f"**Chunk {i}:**")
                st.write(doc.page_content)
                st.divider()

else:
    st.warning("⬅️ Please upload a PDF file to begin.")