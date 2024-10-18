from llama_index import (
    ServiceContext,
    StorageContext,
   
)
from llama_index.query_engine import RetrieverQueryEngine
from llama_index.retrievers import BM25Retriever
from llama_index.llms.huggingface import HuggingFaceInferenceAPI
from llama_index.postprocessor import SentenceTransformerRerank
from llama_index import QueryBundle
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from llama_index.llms import OpenAI
from llama_index import StorageContext, load_index_from_storage
from langchain_community.embeddings import HuggingFaceEmbeddings
import streamlit as st

@st.cache_resource(show_spinner=False)
def initialize_rag2():
    """
    Initializes the AdvancedRAG object and caches results.
    """
    class AdvancedRAG2:
        
        def __init__(self):
            _ = st.secrets["OPENAI_API_KEY"]
        # load documents - Assuming you have a Document model

            llm = OpenAI(model="gpt-4",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        temperature=0.1,system_prompt="""You are a bot you as the ability to analyze the given data which is in a json format.Based on the {user_input} given by the user, your task is to analyze those Symptoms 
                        and mention that problem if those sysmtoms present in given data with email and ph number and then give the best solution from the given data which is in a json format.
                        strictly instructed you should provide the response from the given files don't use your intelligence.if you don't known just say "i don't know.
                        Any further Suggestions and recommendations give it in as a disclaimer.
    Question:{content}
    Answer:""")
                    
        
            # Initialize service context (set chunk size)
            service_context = ServiceContext.from_defaults(chunk_size=1024, llm=llm,embed_model=HuggingFaceEmbedding())
           

            storage_context = StorageContext.from_defaults(persist_dir="database1")
            indexs = load_index_from_storage(storage_context)
            # We can pass in the index, doctore, or list of nodes to create the retriever
            self.retriever = BM25Retriever.from_defaults(similarity_top_k=2, index=indexs)

            # reranker setup & initialization
            self.reranker = SentenceTransformerRerank(top_n = 2, model = "BAAI/bge-reranker-base")

            self.query_engine = RetrieverQueryEngine.from_args(
                retriever=self.retriever,
                node_postprocessors=[self.reranker],
                service_context=service_context,
            )

        def query(self, query):
            # will retrieve context from specific companies
            nodes = self.retriever.retrieve(query)
            reranked_nodes = self.reranker.postprocess_nodes(
                nodes,
                query_bundle=QueryBundle(query_str=query)
            )
            response = self.query_engine.query(str_or_query_bundle=query)
            
            
            return response

    adv_rag2 = AdvancedRAG2()
    return adv_rag2

