import streamlit as st
import os
import time
import tempfile
from dotenv import load_dotenv

from langchain_nvidia_ai_endpoints import ChatNVIDIA
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_community.vectorstores import FAISS

# ------------------ CONFIG ------------------
load_dotenv()
api_key = os.getenv("NVIDIA_API_KEY")

if not api_key:
    st.error("Missing NVIDIA_API_KEY. Please set it in your .env file.")
    st.stop()

llm = ChatNVIDIA(
    model="meta/llama-3.1-8b-instruct",  
    api_key=api_key, 
    temperature=0.3
)

# ------------------ UI ------------------
st.set_page_config(page_title="Nvidia NIM RAG", layout="wide")
st.title("📄 RAG using Nvidia NIM")

uploaded_files = st.file_uploader(
    "Upload PDF files",
    type="pdf",
    accept_multiple_files=True
)

# ------------------ PROMPT ------------------
prompt = ChatPromptTemplate.from_template("""
You are an AI assistant. Use the context to answer.
Answer ONLY from the provided context.
If the answer is not in the context, say "I don't know".

<context>
{context}
</context>

Question: {input}
""")

# ------------------ VECTOR STORE ------------------
from langchain_community.embeddings import HuggingFaceEmbeddings
def vector_embedding():
    embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
    )
    
    all_docs = []
    for uploaded_file in uploaded_files:
        # Using a temporary file to avoid local directory clutter
        with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            tmp_path = tmp_file.name
        
        loader = PyPDFLoader(tmp_path)
        all_docs.extend(loader.load())
        os.remove(tmp_path) # Clean up after loading

    text_splitter = RecursiveCharacterTextSplitter(chunk_size=700, chunk_overlap=50)
    final_docs = text_splitter.split_documents(all_docs)

    st.session_state.vectors = FAISS.from_documents(final_docs, embeddings)

# ------------------ BUTTON ------------------
if st.button("Create Embeddings", disabled=not uploaded_files):
    with st.spinner("Processing documents..."):
        vector_embedding()
    st.success("Vector Database is Ready!")

# ------------------ QUERY ------------------
user_question = st.text_input("Ask a question from your documents")

if user_question:
    if "vectors" not in st.session_state:
        st.error("Please click 'Create Embeddings' first.")
    else:
        document_chain = create_stuff_documents_chain(llm, prompt)
        retriever = st.session_state.vectors.as_retriever(search_kwargs={"k": 5})
        retrieval_chain = create_retrieval_chain(retriever, document_chain)

        start = time.process_time()
        response = retrieval_chain.invoke({'input': user_question})
        process_time = time.process_time() - start

        st.subheader("Answer")
        st.write(response['answer'])
        st.caption(f"⏱ Response time: {round(process_time, 2)} seconds")

        with st.expander("🔍 Retrieved Context"):
            for i, doc in enumerate(response["context"]):
                st.markdown(f"**Chunk {i+1}:**")
                st.write(doc.page_content)
                st.write("---")