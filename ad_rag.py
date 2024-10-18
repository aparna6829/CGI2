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
def Getting_the_historic_data():
    """
    Initializes the AdvancedRAG object and caches results.
    """
    class AdvancedRAG:
        
        def __init__(self):
            _ = st.secrets["OPENAI_API_KEY"]
        # load documents - Assuming you have a Document model
            
            llm = OpenAI(model="gpt-4",
                        api_key=st.secrets["OPENAI_API_KEY"],
                        temperature=0.1,system_prompt="""You are a bot you as the ability to analyze the given data based on the {email} and {phone_number} given by the user,now  provide the summazied response about that doctor and patient.strictly instructed provide the proper summary related to that email and phone number only.Don't provide the data from outoff the given data.
                        strictly instructed if in conversation doctor or patient name they mention need to diasplay and if any clinical test or data they metion provide that one also.
                        Only mention the response related to that {user_input} only.
                        if the givem email and phone number not in the give data ,just say "the history related to this data not in the given data."
                        strictly instructed you should provide the response from the given files don't use your intelligence.if you don't known just say "i don't known""")
                    
        
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

    adv_rag = AdvancedRAG()
    return adv_rag
