import streamlit as st
import pandas as pd
import numpy as np
import base64
import hashlib
import requests
import io
from PIL import Image
from sentence_transformers import SentenceTransformer

# Constants
API_KEY = st.secrets["OPENAI_API_KEY"]
MODEL_NAME = 'all-MiniLM-L6-v2'
DIMENSION = 384
FAISS_INDEX_PATH = "faiss_index.index"
HASHES_FILE_PATH = "image_hashes.txt"

# st.set_page_config(layout='wide')

# Initialize SentenceTransformer
embedder = SentenceTransformer(MODEL_NAME)

def encode_image(image_file):
    image = Image.open(image_file)
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    return base64.b64encode(buffered.getvalue()).decode('utf-8')

def compute_image_hash(image):
    hasher = hashlib.sha256()
    image.seek(0)
    buf = image.read()
    hasher.update(buf)
    return hasher.hexdigest()

def extract_info_from_image(base64_image, retry=False):
    extract_prompt_template = """
    Please analyze this medical prescription image and extract all relevant information in a detailed paragraph format. 
    Include details such as doctor name, patient name, age, sex, hospital name, diagnosis, medicine with dosages and when to take them, date of prescription, and any other relevant information.
    After the paragraph, provide a list of tags for the extracted information in the following format:
    [DOCTOR_NAME]: <extracted doctor name>
    [PATIENT_NAME]: <extracted patient name>
    [PATIENT_AGE]: <extracted patient age>
    [PATIENT_SEX]: <extracted patient sex>
    [HOSPITAL]: <extracted hospital name>
    [DIAGNOSIS]: <extracted diagnosis>
    [MEDICATIONS]: <extracted medicine with dosages and when to take them>
    [DATE]: <extracted date of prescription>
    [OTHER]: <any other relevant information>
    
    Pay special attention to the [MEDICATIONS] field. Make sure to include all medications, their dosages, and instructions on when to take them if available in the prescription.
    """
    
    if retry:
        extract_prompt_template += """
        This is a second attempt at extraction. Please focus specifically on identifying any medications, their dosages, and instructions on when to take them. If you find any such information, make sure to include it in the [MEDICATIONS] field.
        """
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": extract_prompt_template},
                    {"type": "image_url", "image_url": {"url": f"data:image/jpeg;base64,{base64_image}"}}
                ]
            }
        ],
        "max_tokens": 4096
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code != 200:
        st.error(f"Error from OpenAI API: {response.status_code} - {response.text}")
        return {"paragraph": "", "tags": {}}

    response_json = response.json()
    if 'choices' not in response_json:
        st.error(f"Unexpected API response format: {response_json}")
        return {"paragraph": "", "tags": {}}

    extracted_text = response_json['choices'][0]['message']['content']
    
    # Split the extracted text into paragraph and tags
    parts = extracted_text.split('\n\n', 1)
    paragraph = parts[0]
    tags = {}
    if len(parts) > 1:
        for line in parts[1].split('\n'):
            if ':' in line:
                key, value = line.split(':', 1)
                tags[key.strip('[]')] = value.strip()
    
    return {"paragraph": paragraph, "tags": tags}

def rerun_extraction_if_needed(base64_image, extracted_info):
    if not extracted_info['tags'].get('MEDICATIONS') or extracted_info['tags']['MEDICATIONS'] == "N/A":
        # st.warning("No medication information found in the initial extraction. Attempting to rerun the extraction...")
        return extract_info_from_image(base64_image, retry=True)
    return extracted_info

def display_info_table(info_list):
    st.write("Extracted Information from All Images:")
    
    # Convert the list of dictionaries to a pandas DataFrame
    df = pd.DataFrame([item['tags'] for item in info_list])
    
    # Reorder columns if needed
    desired_order = ['DOCTOR_NAME', 'PATIENT_NAME', 'PATIENT_AGE', 'PATIENT_SEX', 'HOSPITAL', 'DIAGNOSIS', 'MEDICATIONS', 'DATE', 'OTHER']
    df = df.reindex(columns=[col for col in desired_order if col in df.columns] + [col for col in df.columns if col not in desired_order])
    
    # Display the table
    st.table(df)

def query_extracted_info(query, faiss_index, extracted_info_list, distance_threshold=1.0):
    query_embedding = embedder.encode([query])
    
    # st.write(f"Number of vectors in Faiss index: {faiss_index.ntotal}")
    
    if faiss_index.ntotal == 0:
        return "No prescriptions have been indexed yet. Please upload some images first."
    
    D, I = faiss_index.search(np.array(query_embedding, dtype='float32'), k=faiss_index.ntotal)
    
    # st.write(f"Search results - Distances: {D}, Indices: {I}")

    
    relevant_results = []
    for dist, idx in zip(D[0], I[0]):
        if dist <= distance_threshold and idx < len(extracted_info_list):
            relevant_results.append((dist, extracted_info_list[idx]['paragraph']))
    
    if not relevant_results:
        return "No relevant prescription found for the query."
    
    # Sort results by distance (most relevant first)
    relevant_results.sort(key=lambda x: x[0])
    
    # Combine relevant paragraphs
    combined_info = "\n\n".join([result[1] for result in relevant_results])
    
    # Use OpenAI to generate a response based on the query and relevant information
    response = generate_response(query, combined_info)
    
    return response

def generate_response(query, context):
    prompt = f"""
    Given the following context from medical prescriptions:

    {context}

    Please answer the following question:
    {query}

    Provide a concise and relevant answer based only on the information given in the context.
    If the information is not available in the context, please state that it cannot be determined from the given information.
    """
    
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {API_KEY}"
    }
    payload = {
        "model": "gpt-4o",
        "messages": [
            {"role": "system", "content": "You are a helpful assistant that answers questions about medical prescriptions based on given context."},
            {"role": "user", "content": prompt}
        ],
        "max_tokens": 1024
    }
    response = requests.post("https://api.openai.com/v1/chat/completions", headers=headers, json=payload)
    
    if response.status_code != 200:
        st.error(f"Error from OpenAI API: {response.status_code} - {response.text}")
        return "Error: Unable to generate response."

    response_json = response.json()
    return response_json['choices'][0]['message']['content']

