import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import pinecone

class Conversation:
    def __init__(self):
        self.history = []

    def add_message(self, role, message):
        self.history.append((role, message))

    def get_conversation(self):
        conversation_text = ""
        for role, message in self.history:
            conversation_text += f"{role}: {message}\n"
        return conversation_text

st.set_page_config(page_title="Immigration Q&A", layout="wide", initial_sidebar_state="expanded")

# Removed the markdown using 'result' here. It is now placed inside the if condition.

custom_css = """
<style>
    #MainMenu {visibility: hidden;}
    #root > div:nth-child(1) > div > div > div > div > section > div {padding-top: 0rem;}
    footer {visibility: hidden;}
    header {
        visibility: hidden;
        overflow: hidden;
    }
    .stApp {
        background-color: #FFFFE6;
        color: #000;
    }
    .stApp h1 a, .stApp h2 a, .stApp h3 a, .stApp h4 a, .stApp h5 a, .stApp h6 a,
    .stApp h1 svg, .stApp h2 svg, .stApp h3 svg, .stApp h4 svg, .stApp h5 svg, .stApp h6 svg {
        display: none;
    }
    .stApp .st-expander {
        border: 1px solid #000;
        border-radius: 5px;
        padding: 10px;
        margin-bottom: 10px;
    }
</style>
"""

# Rest of your code...

OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]
PINECONE_API_KEY = st.secrets["PINECONE_API_KEY"]
PINECONE_API_ENV = st.secrets["PINECONE_API_ENV"]
embeddings = OpenAIEmbeddings(openai_api_key=OPENAI_API_KEY)

pinecone.init(api_key=PINECONE_API_KEY, environment=PINECONE_API_ENV)
index_name = "immigration"
docsearch = Pinecone.from_existing_index(index_name, embeddings)

with st.form(key="my_form"):
    query = st.text_input("Enter your question:")
    submit_button = st.form_submit_button("Submit")

if query:
    # Create conversation in session_state if it doesn't exist
    if "conversation" not in st.session_state:
        st.session_state.conversation = Conversation()

    # Add the user's message to the conversation history
    st.session_state.conversation.add_message('Human', query)

    # Rest of your code...

    docs = docsearch.similarity_search(query,k=5)
    
    with st.spinner('Processing your question...'):
        #result = conversation.predict(input=prompt)
        result = chain.run(input_documents=docs, question=prompt)

    # Add the AI's response to the conversation history
    st.session_state.conversation.add_message('AI', result)

    #st.header("Prompt")
    #st.write(prompt)  # Display the prompt value

    st.header("Answer")
    st.write(result)  # Display the AI-generated answer

    # Display search results
    st.subheader("Sources")
    desired_indices = [1, 5]
    for idx, index in enumerate(desired_indices):
        if index-1 < len(docs):  # Python uses 0-indexing
            doc = docs[index-1]
            with st.beta_expander(f"Source {idx+1}", expanded=True): 
                st.markdown(doc.page_content)  # Display each desired search result
                #st.write("---")
else:
    st.markdown("""
    ## Answer
    No question has been asked yet.""")
