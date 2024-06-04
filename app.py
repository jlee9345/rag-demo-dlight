import streamlit as st
import openai
import pandas as pd
from langchain.vectorstores import Chroma
from langchain.embeddings import OpenAIEmbeddings
from langchain.llms import OpenAI
from langchain.chains.question_answering import load_qa_chain
from langchain.prompts import load_prompt

# Function to load data from Excel
def load_data(file):
    df = pd.read_excel(file)
    return df

# Function to create and configure ChromaDB with data
def create_chroma_db(df):
    # Convert dataframe to list of dictionaries for ChromaDB
    documents = df.to_dict(orient='records')
    # Initialize ChromaDB with the data
    embeddings = OpenAIEmbeddings()
    vectorstore = Chroma.from_documents(documents, embeddings)
    return vectorstore

# Streamlit app
st.title('Retrieval Augmented Generation with Langchain')

# Input OpenAI API Key
openai_api_key = st.text_input('OpenAI API Key', type='password')

# File uploader for Excel file
uploaded_file = st.file_uploader('Upload your Excel file', type='xlsx')

if uploaded_file and openai_api_key:
    # Set OpenAI API Key
    openai.api_key = openai_api_key
    
    # Load data from the uploaded Excel file
    data = load_data(uploaded_file)
    st.write('Data from Excel:', data)
    
    # Create ChromaDB with the data
    chroma_db = create_chroma_db(data)
    
    # Create a Langchain QA Chain
    llm = OpenAI(api_key=openai_api_key)
    retriever = chroma_db.as_retriever()
    qa_chain = load_qa_chain(llm, retriever=retriever)
    
    # Input query
    query = st.text_input('Enter your query:')
    
    if query:
        # Generate response using the chain
        response = qa_chain.run(query)
        st.write('Response:', response)
