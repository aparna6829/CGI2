import streamlit as st
import tempfile
import os
import hashlib
import google.generativeai as genai
from langchain_community.document_loaders import TextLoader
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.chains import RetrievalQAWithSourcesChain
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint



# Configure APIs
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
llm = ChatGoogleGenerativeAI(model="gemini-pro", google_api_key=GOOGLE_API_KEY)

# Function to compute hash of the file content
def compute_file_hash(file_path):
    hasher = hashlib.sha256()
    with open(file_path, 'rb') as f:
        buf = f.read()
        hasher.update(buf)
    return hasher.hexdigest()

# Function to get the file paths for cached transcriptions and summaries
def get_cache_file_paths(file_hash):
    transcription_path = os.path.join("cache", f"{file_hash}_transcription.txt")
    summary_path = os.path.join("cache", f"{file_hash}_summary.txt")
    return transcription_path, summary_path

# Function to check if cached files exist and read them
def read_cache(file_hash):
    transcription_path, summary_path = get_cache_file_paths(file_hash)
    transcription = None
    summary = None

    if os.path.exists(transcription_path):
        with open(transcription_path, 'r') as f:
            transcription = f.read()

    if os.path.exists(summary_path):
        with open(summary_path, 'r') as f:
            summary = f.read()

    return transcription, summary

# Function to write to cache files
def write_cache(file_hash, transcription=None, summary=None):
    transcription_path, summary_path = get_cache_file_paths(file_hash)

    if transcription:
        with open(transcription_path, 'w') as f:
            f.write(transcription)

    if summary:
        with open(summary_path, 'w') as f:
            f.write(summary)

# Function to transcribe audio using Google's Generative AI
def transcribe_audio(audio_file_path):
    file_hash = compute_file_hash(audio_file_path)
    transcription, _ = read_cache(file_hash)

    if transcription:
        return transcription

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(["Please transcribe the following audio.", audio_file])
    transcription = response.text
    _, summary = read_cache(file_hash)  # Check if summary exists to avoid overwriting it
    write_cache(file_hash, transcription=transcription, summary=summary)
    return transcription

# Function to summarize audio using Google's Generative AI
def summarize_audio(audio_file_path):
    file_hash = compute_file_hash(audio_file_path)
    _, summary = read_cache(file_hash)

    if summary:
        return summary

    model = genai.GenerativeModel("models/gemini-1.5-pro-latest")
    audio_file = genai.upload_file(path=audio_file_path)
    response = model.generate_content(["Please summarize the following audio.", audio_file])
    summary = response.text
    transcription, _ = read_cache(file_hash)  # Check if transcription exists to avoid overwriting it
    write_cache(file_hash, transcription=transcription, summary=summary)
    return summary

# Function to create embeddings from a file
def create_embeddings(file_path):
    loader = TextLoader(file_path)
    data = loader.load_and_split()
    embeddings = HuggingFaceEmbeddings()
    db = FAISS.from_documents(data, embeddings)
    db.save_local("Audio_Index")

# Function to get response from the embeddings
def get_response_from(embedding_path):
    embeddings = HuggingFaceEmbeddings()
    docsearch = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)
    chain = RetrievalQAWithSourcesChain.from_chain_type(llm, chain_type="stuff", return_source_documents=True, retriever=docsearch.as_retriever())
    return chain

# Function to save uploaded file
def save_uploaded_file(uploaded_file):
    try:
        with tempfile.NamedTemporaryFile(delete=False, suffix='.' + uploaded_file.name.split('.')[-1]) as tmp_file:
            tmp_file.write(uploaded_file.getvalue())
            return tmp_file.name
    except Exception as e:
        st.error(f"Error handling uploaded file: {e}")
        return None

# Create cache directory if it doesn't exist
if not os.path.exists("cache"):
    os.makedirs("cache")
     
     
     
