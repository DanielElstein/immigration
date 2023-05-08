pip install unstructured

import streamlit as st
import openai

# Get OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key

from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter

data = loader.load()

print (f'You have {len(data)} document(s) in your data')
print (f'There are {len(data[0].page_content)} characters in your document')

text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=0)
texts = text_splitter.split_documents(data)
