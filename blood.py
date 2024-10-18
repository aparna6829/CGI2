import os
import tqdm
import tempfile
import streamlit as st
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores.faiss import FAISS
from langchain_community.llms.huggingface_endpoint import HuggingFaceEndpoint
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
 
# from dotenv import load_dotenv
# load_dotenv()
# HUGGINGFACEHUB_API_TOKEN = os.getenv("HUGGINGFACEHUB_API_TOKEN")
# repo_id = "mistralai/Mistral-7B-Instruct-v0.2"
# # repo_id = "facebook/bart-large-cnn"
# llm = HuggingFaceEndpoint(repo_id=repo_id, temperature=0.1, huggingfacehub_api_token=HUGGINGFACEHUB_API_TOKEN, max_new_tokens=4096)

from langchain_google_genai import ChatGoogleGenerativeAI
os.environ["GOOGLE_API_KEY"] = st.secrets["GOOGLE_API_KEY"]
llm = ChatGoogleGenerativeAI(model="gemini-pro")

@st.cache_resource(show_spinner=False)
def load_file(uploaded_file, file_name):
    temp_dir = tempfile.TemporaryDirectory()
    with open(os.path.join(temp_dir.name, uploaded_file.name), "wb") as f:
        f.write(uploaded_file.read())
    files = [os.path.join(temp_dir.name, filename) for filename in os.listdir(temp_dir.name)]
    documents = []
    for file in tqdm.tqdm(files):
        try:
            if file.lower().endswith(".pdf"):
                loader = PyPDFLoader(file)
                data = loader.load_and_split()
            else:
                continue
            documents.extend(data)
        except Exception as error:
            print(error)
            raise

    embeddings = HuggingFaceEmbeddings()
    vector=FAISS.from_documents(data,embedding=embeddings)
    vector.save_local(f"{file_name}_INDEX")
    print("FAISS DB SAVED")
    return documents

@st.cache_resource(show_spinner=False)
def get_summarized_response(embedding_path, file_upload_name):
    cache_key = f"embeddings_{embedding_path}_{file_upload_name}"
    chain = None
    if cache_key not in st.session_state:
        embeddings = HuggingFaceEmbeddings()
        vector = FAISS.load_local(embedding_path, embeddings, allow_dangerous_deserialization=True)
        db=vector.as_retriever()
        template = """You are a bot who is expert in analyzing and inferencing medical reports, blood reports, scanned reports.
                    You also have the ability to summarize the reports and respond to the user queries as required.
        <context>
        {context}
        </context>
        
        Question: {input}
        """
                    
        prompt = ChatPromptTemplate.from_template(template=template)
        # Create a chain
        doc_chain = create_stuff_documents_chain(llm, prompt)
        chain = create_retrieval_chain(db, doc_chain)
        st.session_state[cache_key] = chain
    else:
        # If embeddings are cached, retrieve them from the session state
        chain = st.session_state[cache_key]
    return chain

    