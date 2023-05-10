import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

hide_footer_style = """
<style>
    .viewerbadge-container {
        visibility: hidden;
    }
    footer {
        visibility: hidden;
    }
    footer:after {
        content: 'goodbye';
        visibility: visible;
        display: block;
        position: relative;
        padding: 5px;
        top: 2px;
    }
</style>
"""

st.markdown(hide_footer_style, unsafe_allow_html=True)



OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

# Initialize Pinecone
pinecone.init(
    api_key=PINECONE_API_KEY,
    environment=PINECONE_API_ENV
)
index_name = "langchaintest2"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

st.title("Immigration Search")
query = st.text_input("Enter your query:")

if query:
    docs = docsearch.similarity_search(query, include_metadata=True)

    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    result = chain.run(input_documents=docs, question=query)

    st.header("Search Results and Answers")
    st.write(result)

