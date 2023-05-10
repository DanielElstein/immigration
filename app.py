import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

# Move st.set_page_config() before any other Streamlit command
st.set_page_config(page_title="Immigration Q&A", layout="wide", initial_sidebar_state="expanded")

hide_menu_style = """
        <style>
        #MainMenu {visibility: hidden;}
        footer {visibility: hidden;}
        header {visibility: hidden;}
        .stApp {
            background-color: #ADD8E6;
        }
        .anchor svg {
            display: none;
        }
        /* Style the input box */
    .stTextInput input {
        padding: 10px;
        border: 1px solid #ccc;
        border-radius: 5px;
    }
    
    /* Style the submit button */
    button[data-baseweb="button"] {
        background-color: #4CAF50;
        border: none;
        color: white;
        padding: 10px 20px;
        text-align: center;
        text-decoration: none;
        display: inline-block;
        font-size: 16px;
        margin: 4px 2px;
        cursor: pointer;
        border-radius: 5px;
    }
    
    /* Style the headers */
    h1, h2 {
        font-family: Arial, sans-serif;
    }
    
    /* Style the body text */
    body, p {
        font-family: Arial, sans-serif;
        color: #333;
    }
        </style>
        """

st.markdown(hide_menu_style, unsafe_allow_html=True)






# Add JavaScript code to handle Enter key press
enter_key_js = """
<script>
    document.addEventListener("DOMContentLoaded", function(event) {
        var input = document.querySelector("input");
        input.addEventListener("keyup", function(e) {
            if (e.key === "Enter") {
                e.preventDefault();
                setTimeout(function() {
                    var button = document.querySelector("button[data-baseweb='button']");
                    if (button) {
                        button.click();
                    }
                }, 100);
            }
        });
    });
</script>
"""
st.markdown(enter_key_js, unsafe_allow_html=True)

image_path = 'liberty.png'
st.image(image_path, width=300)

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
# Create a form
with st.form(key="my_form"):
    query = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Submit")

# Check if the form is submitted
if submit_button:
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
