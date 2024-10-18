from ad_rag import Getting_the_historic_data
from agent import analyze_symptoms, extract_tables
from ad_rag2 import initialize_rag2
# from sharepoint_rpa import handle_prompt,handle_confirmation
from audio_processing import transcribe_audio,summarize_audio,save_uploaded_file,create_embeddings,get_response_from
import streamlit as st
import pandas as pd
import os
from langchain.prompts import PromptTemplate
from langchain_google_genai import ChatGoogleGenerativeAI
import tempfile
import os
import google.generativeai as genai
# from blood import extract_text_from_pdf,save_text,get_textfile_from,get_summarized_response
from scan_utils import call_gpt4_model_for_analysis,chat_eli
from scan_utils import call_gpt4_model_for_analysis, scan_prompt, retina_prompt,blood
from blood1 import get_response,process_pdf
from youtubesearchpython import VideosSearch
from youtube import search_youtube_videos
from langchain.globals import set_llm_cache
import hashlib
from gptcache import Cache
from gptcache.manager.factory import manager_factory
from gptcache.processor.pre import get_prompt
from langchain.cache import GPTCache
from blood import load_file
from blood import get_summarized_response
import base64
from medical import rag
import re
import faiss
from medical_p import compute_image_hash
from medical_p import encode_image, extract_info_from_image, rerun_extraction_if_needed, display_info_table, query_extracted_info
import numpy as np
from sentence_transformers import SentenceTransformer


# Initialize SentenceTransformer
MODEL_NAME = 'all-MiniLM-L6-v2'
DIMENSION = 384
embedder = SentenceTransformer(MODEL_NAME)

def extract_contact_information(result):
    headings = [
        "Introduction",
        "CPT Code with Description and Cost",
        "Detailed Information",
        "Additional Details",
        "Total Cost",
        "Conclusion"
    ]
    contact_information = {}
 
    # Create a more flexible pattern to match headings
    heading_patterns = [rf"###\s*\d+\.\s*\**{re.escape(heading)}\**" for heading in headings]
 
    # Find all headings in the result text
    heading_matches = []
    for pattern in heading_patterns:
        heading_matches.extend(re.finditer(pattern, result, re.IGNORECASE))
 
    # Sort the heading matches by their start position
    heading_matches.sort(key=lambda match: match.start())
 
    # Extract the content between the headings
    for i in range(len(heading_matches)):
        current_heading = heading_matches[i]
        heading = re.sub(r"###\s*\d+\.\s*\**|\**", "", current_heading.group(), flags=re.IGNORECASE).strip()
       
        if i < len(heading_matches) - 1:
            next_heading = heading_matches[i + 1]
            contact_info = result[current_heading.end():next_heading.start()].strip()
        else:
            contact_info = result[current_heading.end():].strip()
       
        contact_information[heading] = contact_info
 
    return contact_information
 
def display_contact_information(contact_information):
    for heading, contact_info in contact_information.items():
        st.write(f"## {heading}")
        with st.expander("Show More Information", expanded=False):
            st.write(contact_info.strip(), unsafe_allow_html=True)
def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()
 
def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )
set_llm_cache(GPTCache(init_gptcache))

st.set_page_config(page_title="Diagnosis Bot",
page_icon="🩺",
layout="wide")
# Configure Google API for audio summarization
GOOGLE_API_KEY = st.secrets["GOOGLE_API_KEY"]
genai.configure(api_key=GOOGLE_API_KEY)
# Custom CSS
def load_css():
    with open("static/styles.css", "r") as f:
        css = f"<style>{f.read()}</style>"
        st.markdown(css, unsafe_allow_html=True)

hide_st_style = """
    <style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    </style>
    """
st.markdown(hide_st_style, unsafe_allow_html=True)
# st.markdown(f'''     
#     <style>section[data-testid="stSidebar"] .css-ng1t4o {{width: 14rem;}}</style>
#     ''', unsafe_allow_html=True)
st.markdown(f'''
    <style>.st-emotion-cache-1avcm0n{{visibility: hidden}}</style>
    ''', unsafe_allow_html=True)
st.markdown("""
    <style>.st-emotion-cache-vj1c9o {background-color:rgb(38 39 48 / 0%);}</style>
    """, unsafe_allow_html=True)

load_css()

####Header and logo
def img_to_base64(img_path):
    with open(img_path, "rb") as img_file:
        return base64.b64encode(img_file.read()).decode()



# Path to your image
img_path = "static/CGI_logo.png"
img_base64 = img_to_base64(img_path)

# Create header container
header = st.container()
header.write(f"""
    <div class='fixed-header'>
        <img src="data:image/png;base64,{img_base64}" class="logo">
        <h1>Efficient Clinical Review Operations AI</h1>
    </div>
""", unsafe_allow_html=True)

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)

# import base64
# image_path = "static/logo_promptora.png"
# with open(image_path, "rb") as f:
#     image_bytes = f.read()
# image_base64 = base64.b64encode(image_bytes).decode()

# div = f"""
#     <div class="watermark">
#         <img src="data:image/jpeg;base64,{image_base64}" width=35 height=25>
#     </div>
# """
# st.markdown(div, unsafe_allow_html=True)
st.markdown("""
    <style>
    /* Customize tab container */
    .stTabs [role="tablist"] {
        border-bottom: 2px solid #rgb(210 210 210 1);
        border:"none";
        border-width: 1px;
        background-color:rgba(255, 255, 255, 0);
        
    }

    /* Customize individual tabs */
    .stTabs [role="tab"] {
        padding: 10px;
        font-size: 18px;
        color: #151515;
        letter-spacing: 1.3px;
        font-weight:800 !important;
        border-bottom: 10px;
        margin:0px;
        font-family:'Source Sans Pro' !important;
        font-style:Semibold;
    }
    
    /* Customize active tab */
    .stTabs [role="tab"][aria-selected="true"] {
        background color:blue;
        color: #5236AB;
        font-size: 20px;
        border-bottom: 4px solid #5236AB;
        border-style: solid;
      
    }

    /* Customize tab content */
    .stTabs [role="tabpanel"] {
        padding: 10px;
        # border: 0px solid #3eacda;
        border-top: none;
        border-radius: 0 0 5px 5px;
        background: #ffffff;
    }
    
    .stTabs [role="tab"][aria-selected="true"]:hover{
        border-bottom:'none'
        color:red;
    }
    .st-c2{
        background-color: #0000;
    }


    </style>
    """, unsafe_allow_html=True)
def get_hashed_name(name):
    return hashlib.sha256(name.encode()).hexdigest()

def init_gptcache(cache_obj: Cache, llm: str):
    hashed_llm = get_hashed_name(llm)
    cache_obj.init(
        pre_embedding_func=get_prompt,
        data_manager=manager_factory(manager="map", data_dir=f"map_cache_{hashed_llm}"),
    )

set_llm_cache(GPTCache(init_gptcache))

def main():
    tab1, tab2 ,tab3,tab4,tab5, tab6 = st.tabs(["Diagnosis 🩺","Audio Transcription 🔊","Blood Report Analysis 💉🩸","Scan Room 🔬","Medical Procedure Cost Analysis 📈", "Medical Prescriptions 💊" ])
    
    with tab1:
        
        def get_historic_data(combined_query):
            adv_rag = Getting_the_historic_data()
            result = adv_rag.query(combined_query)
            st.session_state.response = result.response
            st.session_state.historical_data_processed = True

        def process_user_input(user_input):
            st.session_state.response_data = analyze_symptoms(user_input)
            response, tables, scholar_links  = st.session_state.response_data
 
            st.session_state.internet_response = response
            st.session_state.tables = tables
            st.session_state.scholar_links = scholar_links
            # st.session_state.youtube_links = ytlinks

        if 'response' not in st.session_state:
            st.session_state.response = None

        if 'response2' not in st.session_state:
            st.session_state.response2 = None

        if 'historical_data_processed' not in st.session_state:
            st.session_state.historical_data_processed = ""

        if 'recommendations_processed' not in st.session_state:
            st.session_state.recommendations_processed = False

        if 'response_data' not in st.session_state:
            st.session_state.response_data = None

        if 'email' not in st.session_state:
            st.session_state.email = None

        if 'phone_number' not in st.session_state:
            st.session_state.phone_number = None
        
        if 'uploaded_file_diagnosis' not in st.session_state:
            st.session_state.uploaded_file_diagnosis = None
            
        if 'user_input_diagnosis' not in st.session_state:
            st.session_state.user_input_diagnosis = ""
        if 'chain_db' not in st.session_state:
            st.session_state.chain_diag = None
        if "response_blood_db" not in st.session_state:
            st.session_state.response_blood_db = ""
        if 'internet_response' not in st.session_state:
            st.session_state.internet_response = None

        if 'tables' not in st.session_state:
            st.session_state.tables = None

        if 'scholar_links' not in st.session_state:
            st.session_state.scholar_links = None
            
        if 'youtube_links' not in st.session_state:
            st.session_state.youtube_links = None
            
        if 'current_step' not in st.session_state:
            st.session_state.current_step = None
        
        i = 0
        j = 1
        
        user_query_1 = st.text_input("Enter your Email ", key=f"user_input_{i}")
        user_query_2 = st.text_input("Enter your Phone number: ", key=f"user_input_{j}")
        combined_query = f"{user_query_1} {user_query_2}"

        if st.button("Submit"):
            
            with st.spinner("Getting Historical Data"):
                   # Clear all session states
                st.session_state.clear()

                # Reset the session states needed for the initial steps
                st.session_state.email = user_query_1
                st.session_state.phone_number = user_query_2
                st.session_state.current_step = None

                if user_query_1 and user_query_2:
                    st.session_state.email = user_query_1
                    st.session_state.phone_number = user_query_2


                    get_historic_data(combined_query)
                   
        if st.session_state.historical_data_processed:
        
            with st.expander("Historical Data"):
                st.write(st.session_state.response)
        
           
                # Create columns for layout
            col1, col2 = st.columns([2,15])

            # Use the first column for the file uploader
            with col1:
                st.session_state.uploaded_file_diagnosis = st.file_uploader("Upload your PDF file", type="pdf", key="upload", label_visibility="collapsed")

            # Use the second column for the user input
            with col2:
                st.session_state.user_input_diagnosis = st.chat_input("Enter your query:", key="text_input")
                
                

            # Check if a file has been uploaded and handle it separately
            if st.session_state.uploaded_file_diagnosis is not None:
                # Save the uploaded file temporarily
                with open("temp.pdf", "wb") as f:
                    f.write(st.session_state.uploaded_file_diagnosis.getbuffer())

                st.success("File uploaded successfully.")
                with st.spinner("Processing"):
                    # Process the PDF and get the response
                    chain, db = process_pdf("temp.pdf")
                    if st.session_state.user_input_diagnosis:
                        response = get_response(chain, st.session_state.user_input_diagnosis, db)

                        with st.expander("Reponse"):
                            st.write(response)
                   
            else:
                
                # Check if user has entered a chat input and handle it separately
                if st.session_state.user_input_diagnosis:
                    user_input = st.session_state.user_input_diagnosis
                    st.session_state.current_step = "internet_data"
                    with st.spinner("Getting Internet Data"):
                        process_user_input(user_input)

                if st.session_state.current_step == "internet_data" and st.session_state.internet_response:
                    response = st.session_state.internet_response

                    with st.expander("Response from Internet"):
                        st.write(response)

                    if st.session_state.tables and st.session_state.scholar_links:
                        col1, col2 = st.columns(2)
                        with col1:
                            df = pd.DataFrame(st.session_state.tables)
                            with st.spinner("Getting Table Data"):
                                with st.expander("Past 10 Years Data"):
                                    st.dataframe(df, height=200, width=500)
                        with col2:
                            with st.expander("From Google Scholar link"):
                                for link in st.session_state.scholar_links:
                                    st.markdown(link)

                    if st.session_state.scholar_links:
                        with st.spinner("Getting Youtube Links"):
                        # st.title("YouTube Videos Related to Diagnosis")
                            with st.expander("YouTube Links"):
                                user_input = st.session_state.user_input_diagnosis
                                if user_input:
                                    ytlinks = search_youtube_videos(user_input)
                                    st.markdown(ytlinks)

                    if st.session_state.internet_response:
                        with st.spinner("Recommendations from the Historical Data"):
                            adv_rag2 = initialize_rag2()
                            st.session_state.response2 = adv_rag2.query(st.session_state.user_input_diagnosis)

                            with st.expander("Recommendations from the Historical Data"):
                                st.write(st.session_state.response2.response)
    with tab2:
        
        # Initialize session state variables
        if "transcribed_text" not in st.session_state:
            st.session_state.transcribed_text = ""
        if "summarized_text" not in st.session_state:
            st.session_state.summarized_text = ""
        if "chain_audio" not in st.session_state:
            st.session_state.chain_audio = None
        if 'user_input_audio' not in st.session_state:
            st.session_state.user_input_audio = ""
        if "response_audio" not in st.session_state:
            st.session_state.response_audio = ""
        # Streamlit app interface
        col1, col2 = st.columns([2,15])
        with col1:
            audio_file = st.file_uploader("Upload Audio File", type=['wav', 'mp3'],label_visibility='collapsed')


            if audio_file is not None:
                audio_path = save_uploaded_file(audio_file)  # Save the uploaded file and get the path
                st.audio(audio_path)
                st.success("File uploaded successfully.")
            
                if st.button('Summarize Audio'):
                    with st.spinner('Summarizing...'):
                        st.session_state.transcribed_text = transcribe_audio(audio_path)
                        st.session_state.summarized_text = summarize_audio(audio_path)
                        # Compute the file hash to get the cache file paths
                       
                        create_embeddings("Response.txt")
                        st.session_state.chain_audio = get_response_from("Audio_Index")
        
        # Display transcribed and summarized text if available
        if st.session_state.transcribed_text:
            with st.expander("Transcribed Text"):
                st.info(st.session_state.transcribed_text)
        if st.session_state.summarized_text:
            with st.expander("Summarized Text"):
                st.info(st.session_state.summarized_text)
            with col2:
                # Handle user query input and display response
                st.session_state.user_input_audio = st.chat_input("Enter your query here:",key="audio_input")
                with st.spinner("Processing"):
                   
                    if st.session_state.user_input_audio and st.session_state.chain_audio:
                        result = st.session_state.chain_audio.invoke({"question": st.session_state.user_input_audio})
                        st.session_state.response_audio = result["answer"].replace('\n', ' ')
                    
            if st.session_state.response_audio:
                with st.expander("Response"):
                    st.write(st.session_state.response_audio)
            

    with tab3:
        col1, col2 = st.columns([2,15])
        with col1:

            uploaded_file = st.file_uploader("Upload your file ", type=["pdf"], label_visibility="collapsed")
            if uploaded_file is not None:
                st.success("File uploaded successfully.")
                file_name = uploaded_file.name.split('.')[0]

                load_file(uploaded_file,file_name)
                with st.spinner("Processing"):
                    chain = get_summarized_response(f"{file_name}_INDEX", uploaded_file.name)
        with col2:
        
            user_input=st.chat_input("Enter your input here:", key="main_check")
            if user_input:
                with st.spinner("Getting Response"):
                    response = chain.invoke({"input":user_input})
                    # Print the response
                    print(response['answer'])
                    with st.expander("Response"):
                        st.write(response['answer'])
        



    with tab4:
        
        if "user_input_scan" not in st.session_state:
            st.session_state.user_input_sacn = ""
        if 'uploaded_file_scan' not in st.session_state:
            st.session_state.uploaded_file = None
        if 'scan_room' not in st.session_state:
            st.session_state.scan_room = None
     
        if "response_scan" not in st.session_state:
            st.session_state.response_scan = ""
        
        topic = st.selectbox("Select the topic for analysis:", ["Brain Tumour", "Retina","Blood report image Analysis"])

# Set the appropriate prompt based on the selected topic
        if topic == "Brain Tumour":
            selected_prompt = scan_prompt
        elif topic == "Retina":
            selected_prompt = retina_prompt
        elif topic == "Blood report image Analysis":
            selected_prompt = blood
      
            
        st.session_state.uploaded_file_scan = st.file_uploader("Upload an Image", type=["jpg", "jpeg", "png"],label_visibility='collapsed')
        
        # Temporary file handling
        if st.session_state.uploaded_file_scan  is not None:
            with tempfile.NamedTemporaryFile(delete=False, suffix=os.path.splitext(st.session_state.uploaded_file_scan .name)[1]) as tmp_file:
                tmp_file.write(st.session_state.uploaded_file_scan .getvalue())
                st.session_state['filename'] = tmp_file.name
            with st.expander("Image"):
                st.image(st.session_state.uploaded_file_scan,width=100)
            with st.spinner("Processing"):
                if 'filename' in st.session_state and os.path.exists(st.session_state['filename']):
                    st.session_state.scan_room = call_gpt4_model_for_analysis(st.session_state['filename'],selected_prompt)
                  
                if st.session_state.scan_room:
                    with st.expander("Response"):
                        st.markdown(st.session_state.scan_room)
                    css='''
                        <style>
                            [data-testid="stExpander"] div:has(>.streamlit-expanderContent) {
                                overflow: scroll;
                                height: 200px;
                            }
                        </style>
                                    '''

                    st.markdown(css, unsafe_allow_html=True)
                    os.unlink(st.session_state['filename'])
                    
    with tab5:
        col1,col2 = st.columns([2,15])
        with col1:
            st.success("Data Loaded Successfully.")
        with col2:
            user_input = st.chat_input("Enter your query here:", key = "medical_procedure_cost")
            if user_input is not None:
                with st.spinner("Loading..."):
                    ai_response = rag(user_input)
                    result=ai_response['result']
                    contact_information = extract_contact_information(result)
                    display_contact_information(contact_information)
                    # st.write(result)
    
    with tab6:
        if 'extracted_info_list' not in st.session_state:
            st.session_state['extracted_info_list'] = []
        
        if 'faiss_index' not in st.session_state:
            st.session_state['faiss_index'] = faiss.IndexFlatL2(DIMENSION)
        
        if 'image_hashes' not in st.session_state:
            st.session_state['image_hashes'] = set()

        # with st.sidebar:
        # st.header("Upload Prescription Images")
        col1,col2 = st.columns([2,15])
        with col1:
            uploaded_files = st.file_uploader("Upload your prescriptions:", type=["png", "jpg", "jpeg"], accept_multiple_files=True, label_visibility="collapsed")

        if uploaded_files:
            for image_file in uploaded_files:
                image_hash = compute_image_hash(image_file)
                
                if image_hash in st.session_state['image_hashes']:
                    st.warning(f"The image '{image_file.name}' already exists in the database. Skipping.")
                    continue

                with st.expander(f"Uploaded Image: {image_file.name}"):
                    st.image(image_file)
                
                base64_image = encode_image(image_file)
                extracted_info = extract_info_from_image(base64_image)
                
                # Rerun extraction if needed
                extracted_info = rerun_extraction_if_needed(base64_image, extracted_info)

                embedding = embedder.encode([extracted_info['paragraph']])
                st.session_state['faiss_index'].add(np.array(embedding, dtype='float32'))

                st.session_state['extracted_info_list'].append(extracted_info)
                st.session_state['image_hashes'].add(image_hash)

        if st.session_state['extracted_info_list']:
            with st.expander("Extracted_List"):
                display_info_table(st.session_state['extracted_info_list'])

        with col2:
            query = st.chat_input("Enter your query here:", key="Medical_prescription_Analysis")
        if query:
            result = query_extracted_info(query, st.session_state['faiss_index'], st.session_state['extracted_info_list'])
            with st.expander("Query_Result"):
                st.write(result)

if __name__=='__main__':
    main()