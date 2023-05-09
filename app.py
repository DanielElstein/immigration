import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

PINECONE_API_ENV = 'us-west1-gcp-free'
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Get OpenAI API key
openai_api_key = st.secrets["OPENAI_API_KEY"]
openai.api_key = openai_api_key
pinecone.api_key = pinecone_api_key

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,  # find at app.pinecone.io
    environment=PINECONE_API_ENV  # next to api key in console
)
index_name = "langchaintest2"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

st.title("Document Search")
query = st.text_input("Enter your query:")

if query:
    docs = docsearch.similarity_search(query, include_metadata=True)
    st.header("Search Results")
    for idx, doc in enumerate(docs):
        st.subheader(f"Result {idx + 1}:")
        st.write(doc.page_content[:250])
