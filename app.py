import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
from langchain.chains.qa_with_sources import load_qa_with_sources_chain
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

    # Retrieve the conversation history from the session state
    conversation_text = st.session_state.conversation.get_conversation()

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")

    # Perform document search
    docs = docsearch.similarity_search(query, include_metadata=True, k=3)

    
    # Initialize memory
    memory = ConversationBufferMemory()

    # Update memory with conversation_text
    memory.update(conversation_text)

    # Load the question-answering chain
    chain = load_qa_with_sources_chain(llm, chain_type="stuff", memory=memory)  # Replace "stuff" with the actual chain type

    from langchain.prompts import (
    ChatPromptTemplate, 
    MessagesPlaceholder, 
    SystemMessagePromptTemplate, 
    HumanMessagePromptTemplate
    )
    from langchain.chains import ConversationChain
    from langchain.memory import ConversationBufferMemory

    # Define a prompt template
    prompt = ChatPromptTemplate.from_messages([
        SystemMessagePromptTemplate.from_template("Play the role of a friendly immigration lawyer. Respond to questions in detail, in the same language as the human's most recent question. If they ask a question in Spanish, you should answer in Spanish. If they ask a question in French, you should answer in French. And so on, for every language."),
        MessagesPlaceholder(variable_name="history"),
        HumanMessagePromptTemplate.from_template("{input}")
    ])

    # Initialize memory
    memory = ConversationBufferMemory(return_messages=True)

    # Initialize the conversation chain
    conversation_chain = ConversationChain(memory=memory, prompt=prompt, llm=llm)

    # Perform document search
    docs = docsearch.similarity_search(query, include_metadata=True, k=3)

    # Add search results to the memory
    for doc in docs:
        memory.add_message("System", f"Document Content: {doc.page_content}")

    # Add the user's message to the memory
    memory.add_message('Human', query)

    # Run the conversation chain
    with st.spinner('Processing your question...'):
        result = conversation_chain.run(input=query)

    # Add the AI's response to the conversation history
    memory.add_message('AI', result)

    # Display the AI-generated answer
    st.header("Answer")
    st.write(result)

    # Display search results
    if docs:
        st.header("Search Results")
        st.write(f"Total search results: {len(docs)}")  # Display the number of results

        for index, doc in enumerate(docs, 1):
            st.write(f"Result {index}:")
            st.write(doc.page_content[:2000])  # Display the first 2000 characters of each search result
            st.write("---")
    else:
        st.write("No results found.")
