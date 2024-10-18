import os
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from youtubesearchpython import VideosSearch
from langchain_openai import ChatOpenAI
import streamlit as st
import re
import pandas as pd

def extract_tables(lines):
    metrics_data = []
    for line in lines:
        if "|" in line:  # assuming tables are in markdown table format
            metrics_data.append(line)
    return metrics_data

def extract_scholar_links(text):
    # Regular expression to find URLs
    url_regex = r"https://scholar.google.com/[^\s]+"
    scholar_links = re.findall(url_regex, text)
    # print(scholar_links)
    return scholar_links



@st.cache_resource(show_spinner=False)
def analyze_symptoms(user_input):
    
 
    llm=ChatOpenAI(model="gpt-4o",api_key=st.secrets["OPENAI_API_KEY"])
    template = """
     You are a bot capable of analyzing the given {user_input} and corresponding medical symptoms. Your tasks include:

    1.Analyze Symptoms and Provide Solutions: Examine the provided symptoms and offer the best solution available from Google in 300 words.

    2.Provide Historical Data: For every conversation, present past ten years' historical data related to the {user_input} in a tabular format, including causes and numerical figures of people affected.

    3.Provide Article Links: Analyze the symptoms from the {user_input} and provide the best article links exclusively from "https://scholar.google.com/" using {llm}. If you can't find relevant articles, simply say, "No related articles available in the internet" Make sure to only provide working links with related data from "https://scholar.google.com/". If no links are available, say, "The article for these symptoms is not available on the internet."

    4.Any further Suggestions and recommendations give it in as a disclaimer.
    Question:{content}
    Answer:
    """

    prompt = PromptTemplate(template=template, input_variables=["content", "user_input"])
    chain = LLMChain(prompt=prompt, llm=llm)

    response = chain.invoke({"content": user_input, "user_input": user_input,"llm":llm})
    result_lines = response['text'].splitlines()
    # Extract tables and scholar links
    tables = extract_tables(result_lines)
    scholar_links = extract_scholar_links(response['text'])
    
 
    # Remove tables and scholar links from the response
    table_heading = "Historical Data on Treatments for Bipolar Disorder (Past Ten Years)"
    scholar_heading = "Scholarly Article Links"
    
    # Remove tables, scholar links, and headings from the response
    response_text = response['text']
    for table in tables:
        response_text = response_text.replace(table, "")
    for link in scholar_links:
        response_text = response_text.replace(link, "")
    response_text = response_text.replace(table_heading, "").replace(scholar_heading, "")

    return response_text, tables, scholar_links


    
