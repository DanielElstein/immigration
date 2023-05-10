import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Move st.set_page_config() before any other Streamlit command
st.set_page_config(page_title="Immigration Q&A", page_icon=":guardsman:", layout="wide", initial_sidebar_state="expanded")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        body {
            background-color: #1E90FF;
        }
        </style>
        """
st.markdown(hide_menu_style, unsafe_allow_html=True)

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

st.title("Immigration Q&A")
query = st.text_input("Enter your question:")

template = """
Lawyer: Hello! I am your friendly immigration lawyer. How can I assist you today?

Human: {query}

Lawyer: """

if query:
    prompt = template.format(query=query)
    docs = docsearch.similarity_search(query, include_metadata=True)

    from langchain.llms import OpenAI
    from langchain.chains.question_answering import load_qa_chain

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    chain = load_qa_chain(llm, chain_type="stuff")

    with st.spinner('Processing your question...'):
        result = chain.run(input_documents=docs, question=prompt)

    st.header("Answer")
    st.write(result)
