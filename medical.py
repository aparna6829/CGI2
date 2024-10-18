import os
from tqdm import tqdm
from langchain_community.document_loaders import PyPDFLoader, Docx2txtLoader, TextLoader,UnstructuredExcelLoader, CSVLoader
from langchain_community.embeddings import HuggingFaceEmbeddings,OpenAIEmbeddings
from langchain.embeddings import CacheBackedEmbeddings
from langchain_community.vectorstores import FAISS
from langchain.storage import LocalFileStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.chains import RetrievalQA,LLMChain
from langchain.chains.combine_documents.stuff import StuffDocumentsChain
from langchain.prompts import PromptTemplate
from langchain.llms.llamacpp import LlamaCpp
import streamlit as st
import os
import time
from langchain_community.chat_models import ChatOpenAI

files_directory ="llama"

 


files = [os.path.join(files_directory, file) for file in os.listdir(files_directory)]

# Load documents
documents = []

for file in tqdm(files):
    try:
        if file.lower().endswith(".pdf"):
            loader = PyPDFLoader(file)
            data = loader.load()
        elif file.lower().endswith((".doc", ".docx")):
            loader = Docx2txtLoader(file)
            data = loader.load()
        elif file.lower().endswith(".txt"):
            loader = TextLoader(file)
            data = loader.load()
        elif file.lower().endswith(".xlsx"):
            loader = UnstructuredExcelLoader(file)
            data= loader.load()
        elif file.lower().endswith(".csv"):
            loader = CSVLoader(file)
            data = loader.load()
        else:
            continue

        documents.extend(data)
    except Exception as e:
        print(f"Error loading file {file}: {e}")
        continue

# Text splitting
text_splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=200)
esops_documents = text_splitter.transform_documents(documents)
print(f"Number of chunks in documents: {len(esops_documents)}")

# Create VectorStore
store = LocalFileStore("./cache_quar-mistral/")

core_embeddings_model=HuggingFaceEmbeddings()
embedder = CacheBackedEmbeddings.from_bytes_store(core_embeddings_model, store)
vectorstore = FAISS.from_documents(esops_documents, embedder)



model_kwargs="cpu"

prompt_template = """
As a bot, I possess expert knowledge and capabilities to thoroughly comprehend documents of any format, particularly unstructured Excel documents.
Additionally, I specialize in understanding medical terminology and codes, encompassing conceptual, medical, scientific names, and their meanings. 
When presented with a user query, I meticulously analyze the document and its description to ensure clear understanding.
Even if the user provides CPT codes, I am capable of providing accurate procedure code descriptions without fabricating information or engaging in speculation.
Moreover, you have the ability to provide two additional CPT codes that are 50% accurate to the generated output CPT codes.

{context}

1. **Introduction**
   - Provide a brief introduction to the mentioned procedure, outlining its importance in healthcare in atleast 4-5 lines.
   - Highlight the significance of understanding associated costs for informed decision-making.
   - Emphasize the role of medical expenses in the treatment process.
   - Discuss the potential impact of cost considerations on patient care.

2. **CPT Code with Description and Cost**
   - If applicable, present structured data such as tables.The Cost is in USD.
   - Use bold text for headings and relevant data points.

3. **Detailed Information**
   - **Procedure Name:** [Procedure Name]
   - Provide details about the procedure to be followed, outlining its key aspects.
   - Discuss the necessity of the procedure and its relevance to the patient's condition.
   - Highlight any specific steps or precautions associated with the procedure.
   - Explain how the procedure fits into the overall treatment plan.

4. **Additional Details**
   - Include all miscellaneous costs(USD) beyond the procedure itself, such as medication, hospitalization, and other similar expenses.
   - Find the average days required for the treatment.
   - Calculate the daily charges(USD) based on the average days required and elaborate on the specifics of each cost component, including their associated amounts.
   - [Font: Courier New] Ensure consistency in font for a unified appearance.

5. **Total Cost**
   - Calculate the total cost by aggregating all relevant costs from the CPT code with description and cost section as well as the additional details section.
   - Emphasize the total cost in bold to provide a comprehensive overview of the financial implications.
   - [Font: Courier New] Maintain consistent font style throughout to enhance readability.

6. **Conclusion**
   - Summarize the cost analysis findings and highlight key takeaways regarding the procedure's financial implications.
   - Provide a concise summary of the procedure and its significance.
   - Conclude with pertinent closing remarks or guidance for further understanding.

Question: {question}
Helpful Answer:"""


# Set OpenAI API Key
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

llm = ChatOpenAI(model_name="gpt-4o", temperature=0.2, openai_api_key=OPENAI_API_KEY)


retriever = vectorstore.as_retriever()
QA_CHAIN_PROMPT = PromptTemplate.from_template(prompt_template) # prompt_template defined above
llm_chain = LLMChain(llm=llm, prompt=QA_CHAIN_PROMPT, callbacks=None, verbose=False)
document_prompt = PromptTemplate(
    input_variables=["page_content", "source"],
    template="Context:\ncontent:{page_content}\nsource:{source}",
    )
combine_documents_chain = StuffDocumentsChain(
        llm_chain=llm_chain,
        document_variable_name="context",
        document_prompt=document_prompt,
        callbacks=None,
        verbose=False
    )
qa = RetrievalQA(
        combine_documents_chain=combine_documents_chain,
        retriever=retriever,
        return_source_documents=True,
        verbose=False
    )

 
user_input= "what about you?"
# start_time=time.time()
query =user_input
start_time=time.time()
def rag(query:str)-> str:
    response = qa({"query":query})
    # print (response)
    return response

