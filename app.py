import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import pinecone
import re

st.set_page_config(page_title="Immigration Q&A", layout="wide", initial_sidebar_state="expanded")

st.header("Immigration Q&A")

custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background-color: #ddedee;
    }
    .anchor svg {
        display: none;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Display the text
st.markdown("""

*Legal Disclaimer: This platform is meant for informational purposes only. It is not affiliated with USCIS or any other governmental organization, and is not a substitute for professional legal advice. The answers provided are based on the USCIS policy manual and may not cover all aspects of your specific situation. For personalized guidance, please consult an immigration attorney.*
""")

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "langchaintest2"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

# Create a form
with st.form(key="my_form"):
    query = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Submit")

    # Add JavaScript snippet to submit form on Enter key press
    st.markdown(
        """
        <script>
        document.addEventListener('keydown', function(e) {
            if (e.key === 'Enter') {
                document.querySelector('button[data-baseweb="button"]').click();
            }
        });
        </script>
        """,
        unsafe_allow_html=True
    )    
    
custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    .stApp {
        background-color: #ddedee;
    }
    .anchor svg {
        display: none;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)

# Create an instance of OpenAI
llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

# Create an instance of ConversationChain
conversation = ConversationChain(llm=llm, verbose=True, memory=ConversationBufferMemory())


if query:
    
    template = """You are a chatbot having a conversation with a human.

    {chat_history}
    Human: {human_input}
    Chatbot:"""

    prompt = template(
        input_variables=["chat_history", "human_input"], 
        template=template
    )
    memory = ConversationBufferMemory(memory_key="chat_history")
    


    # Generate the response
    with st.spinner('Processing your question...'):
        result = conversation.predict(input=prompt)

    # Save the AI's response to the conversation memory
    memory.save_context({"input": query}, {"output": result})

    # Display the prompt and the answer
    st.header("Prompt")
    st.write(prompt)  # Display the prompt value

    st.header("Answer")
    st.write(result)  # Display the AI-generated answer
