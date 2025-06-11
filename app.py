# Modified RAG Chatbot using Unstructured + LlamaIndex + Gemini/Groq

import streamlit as st
import os
import tempfile
from dotenv import load_dotenv
from unstructured.partition.pdf import partition_pdf
from llama_index.core import Document, VectorStoreIndex, ServiceContext
from llama_index.embeddings import LangchainEmbedding
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from llama_index.llms import ChatGoogle
from llama_index.core import SimpleDirectoryReader
import google.generativeai as genai
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# Load environment variables
load_dotenv()

# API Keys
gemini_api_key = os.getenv("GOOGLE_API_KEY")
gorq_api_key = os.getenv("GORQ_API_KEY")

if not gemini_api_key or not gorq_api_key:
    st.error("Missing required API keys.")
    st.stop()

genai.configure(api_key=gemini_api_key)

# Session state
if "index" not in st.session_state:
    st.session_state.index = None
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []
if "selected_model" not in st.session_state:
    st.session_state.selected_model = "Gemini"

def extract_text_with_unstructured(pdf_file):
    with tempfile.NamedTemporaryFile(delete=False, suffix=".pdf") as tmp:
        tmp.write(pdf_file.getvalue())
        tmp_path = tmp.name

    elements = partition_pdf(filename=tmp_path)
    os.unlink(tmp_path)
    return [str(el) for el in elements if el.text]

def build_index_from_chunks(text_chunks):
    embeddings = LangchainEmbedding(GoogleGenerativeAIEmbeddings(model="models/embedding-001", google_api_key=gemini_api_key))
    service_context = ServiceContext.from_defaults(embed_model=embeddings)
    documents = [Document(text=t) for t in text_chunks]
    st.session_state.index = VectorStoreIndex.from_documents(documents, service_context=service_context)

def query_with_groq(question, context):
    url = "https://api.groq.com/openai/v1/chat/completions"
    headers = {
        "Authorization": f"Bearer {gorq_api_key}",
        "Content-Type": "application/json"
    }
    data = {
        "model": "gemma2-9b-it",
        "messages": [
            {"role": "system", "content": "Use only the given context to answer."},
            {"role": "user", "content": f"Context: {context}\n\nQuestion: {question}"}
        ],
        "temperature": 0.5,
        "max_tokens": 1000
    }
    try:
        session = requests.Session()
        retries = Retry(total=3, backoff_factor=0.5, status_forcelist=[500, 502, 503, 504])
        session.mount('https://', HTTPAdapter(max_retries=retries))
        response = session.post(url, headers=headers, json=data, timeout=30)
        return response.json().get("choices", [{}])[0].get("message", {}).get("content", "No response")
    except Exception as e:
        return f"Groq API error: {str(e)}"

def answer_query(query):
    if not st.session_state.index:
        st.warning("Upload and process PDF first.")
        return
    retriever = st.session_state.index.as_query_engine()
    context_response = retriever.query(query)

    if st.session_state.selected_model == "Gemini":
        return context_response.response
    else:
        return query_with_groq(query, context_response.response)

# Streamlit UI
st.title("Chat with PDF using Unstructured + LlamaIndex")

with st.sidebar:
    st.header("Settings")
    st.session_state.selected_model = st.radio("Select AI Model:", ["Gemini", "Gorq"], index=0)
    pdf_docs = st.file_uploader("Upload PDF files", accept_multiple_files=True)

    if st.button("Process PDFs"):
        if not pdf_docs:
            st.warning("Please upload at least one PDF.")
        else:
            with st.spinner("Extracting and indexing..."):
                all_text_chunks = []
                for pdf in pdf_docs:
                    chunks = extract_text_with_unstructured(pdf)
                    all_text_chunks.extend(chunks)
                build_index_from_chunks(all_text_chunks)
                st.success("PDFs processed successfully!")

# Display chat history
for msg in st.session_state.chat_history:
    with st.chat_message(msg["role"]):
        st.write(msg["content"])

# Chat input
if prompt := st.chat_input("Ask something about your PDFs"):
    st.session_state.chat_history.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.write(prompt)
    with st.chat_message("assistant"):
        response = answer_query(prompt)
        st.write(response)
        st.session_state.chat_history.append({"role": "assistant", "content": response})

def main():
    st.header("Chat with PDF using AI 🤖")

    with st.sidebar:
        st.title("Menu:")
        st.write("📄 Drag and drop PDF files here (Max 10)")

        st.session_state.selected_model = st.radio(
            "Select AI Model:",
            ["Gemini", "Gorq"],
            index=0 if st.session_state.selected_model == "Gemini" else 1
        )

        pdf_docs = st.file_uploader("Upload your PDF(s)", accept_multiple_files=True)
        if pdf_docs:
            st.write(f"📂 Files selected: {len(pdf_docs)} / 10")

        if pdf_docs and len(pdf_docs) > 10:
            st.error("Maximum 10 PDF files allowed.")
        elif st.button("Submit & Process"):
            if not pdf_docs:
                st.warning("Please upload at least one PDF.")
            else:
                with st.spinner("Processing..."):
                    raw_text = get_pdf_text_with_unstructured(pdf_docs)
                    if not raw_text.strip():
                        st.error("No extractable text found.")
                    else:
                        build_llama_index(raw_text)
                        st.success("PDF processing and indexing complete.")
                        st.session_state.chat_history = []

    for message in st.session_state.chat_history:
        with st.chat_message(message["role"]):
            st.write(message["content"])

    if prompt := st.chat_input(f"Ask a question about your PDF (Model: {st.session_state.selected_model})"):
        st.session_state.chat_history.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)
        with st.chat_message("assistant"):
            response = query_index(prompt)
            st.write(response)
            st.session_state.chat_history.append({"role": "assistant", "content": response})

# Run the app
if __name__ == "__main__":
    main()