import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

try:
    OPENAI_API_KEY = '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'
PINECONE_API_KEY = '0afecd30-9c49-4556-900d-be421b5dcddf'
PINECONE_API_ENV = 'us-west1-gcp-free'
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

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
