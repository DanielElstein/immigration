import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

try:
    openai_api_key = st.secrets["OPENAI_API_KEY"]
    openai.api_key = openai_api_key
    pinecone_api_key = st.secrets["PINECONE_API_KEY"]
    pinecone.api_key = openai_api_key
    pinecone_api_env = st.secrets["PINECONE_API_ENV"]
    pinecone.api_env = pinecone_api_env

    st.write("API keys loaded successfully")

    embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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

except Exception as e:
    st.error(f"An error occurred: {str(e)}")
