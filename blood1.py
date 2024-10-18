# app_logic.py
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain.vectorstores.faiss import FAISS
from langchain.chains.qa_with_sources.retrieval import RetrievalQAWithSourcesChain
from langchain.llms.huggingface_endpoint import HuggingFaceEndpoint
import os
import streamlit as st
import google.generativeai as genai

from langchain_openai import ChatOpenAI
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain.chains import create_retrieval_chain
from langchain_google_genai import ChatGoogleGenerativeAI


from langchain.globals import set_llm_cache
import hashlib
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
 
def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()
 
def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )
set_llm_cache(GPTCache(init_gptcache))
def process_pdf(file_path):
    loader = PyPDFLoader(file_path)
    data = loader.load_and_split()
    GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
    genai.configure(api_key=GOOGLE_API_KEY)
   
    llm = ChatGoogleGenerativeAI(model="gemini-pro")
    embeddings = HuggingFaceEmbeddings()

    vector = FAISS.from_documents(data, embedding=embeddings)
    vector.save_local("Blood_INDEX1")

    db = vector.as_retriever()
    template = """You are a bot who is expert in analyzing and inferencing medical reports, blood reports, scanned reports provided by the user{db}.
    You also have the ability to analyze the report regarding patient having any diseases and then summarize the reports and respond to the {user_input} as required and provide the regular food habits,excersises related to those symtoms and highlight the deficiency of the patient at the end of the response mention as a "Note".
    <context>
    {context}
    </context>

    Question: {input}
    """

    # Create a prompt template
    prompt = ChatPromptTemplate.from_template(template)

    # Create a chain 
    doc_chain = create_stuff_documents_chain(llm, prompt)
    chain = create_retrieval_chain(db, doc_chain)
    
    return chain, db

def get_response(chain, user_input, db):
    response = chain.invoke({"input": user_input, "db": db, "user_input": user_input})
    return response['answer']