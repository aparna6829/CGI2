import streamlit as st
import base64
import os
from openai import OpenAI
import tempfile
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

 
os.environ["OPENAI_API_KEY"] = st.secrets["OPENAI_API_KEY"]
 
client = OpenAI()
 
scan_prompt = """You are a medical practictioner and an expert in analzying medical related images working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".
 
Now analyze the image and answer the above questions in the same structured manner defined above.
Any further Suggestions and recommendations give it in as a disclaimer."""

retina_prompt="""You are an expert for scanning the retina image of the eyes and give the brief about the eye   then give the  scale of the diabetics  the patient is suffering .
Then give the solution like what foods the patients can consume and how to get the treatment for that .
Providing the details of the rating :
A clinician has rated the presence of diabetic retinopathy in each image on a scale of 0 to 4, according to the following scale:
0 - No DR
1 - Mild
2 - Moderate
3 - Severe
4 - Proliferative DR
Your task is to create an analysis system capable of assigning a score based on this scale.
Any further Suggestions and recommendations give it in as a disclaimer."""

blood ="""You are a medical practictioner and an expert in analzying blood report images which is in a written format working for a very reputed hospital. You will be provided with images and you need to identify the anomalies, any disease or health issues. You need to generate the result in detailed manner. Write all the findings, next steps, recommendation, etc. You only need to respond if the image is related to a human body and health issues. You must have to answer but also write a disclaimer saying that "Consult with a Doctor before making any decisions".
 
Now analyze the image and answer the above questions in the same structured manner defined above.
Any further Suggestions and recommendations give it in as a disclaimer."""
# Initialize session state variables
if 'uploaded_file' not in st.session_state:
    st.session_state.uploaded_file = None
if 'result' not in st.session_state:
    st.session_state.result = None
 
def encode_image(image_path):
    with open(image_path, "rb") as image_file:
        return base64.b64encode(image_file.read()).decode('utf-8')
 
def call_gpt4_model_for_analysis(filename: str, sample_prompt:str):
    base64_image = encode_image(filename)
   
    messages = [
        {
            "role": "user",
            "content":[
                {
                    "type": "text", "text": sample_prompt
                    },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{base64_image}",
                        "detail": "high"
                        }
                    }
                ]
            }
        ]
 
    response = client.chat.completions.create(
        model = "gpt-4o",
        messages = messages,
        max_tokens = 1024
        )
 
    return response.choices[0].message.content
 
def chat_eli(sample_prompt=scan_prompt):
    # eli5_prompt = "You have to analyse the given image provide by the user\n"
    messages = [
        {
              
            "role": "user",
            "content":[
                {
                    "type": "text", "text": sample_prompt
                    },
            ]
        }
    ]
 
    response = client.chat.completions.create(
        model="gpt-4o",
        messages=messages,
        max_tokens=1500
    )
 
    return response.choices[0].message.content
 
