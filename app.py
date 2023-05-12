import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
from langchain.memory import ConversationBufferMemory
from langchain.llms import OpenAI
from langchain.chains import ConversationChain
import pinecone

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

# Initialize the conversation memory
memory = ConversationBufferMemory()

if query:
    # Create conversation memory if it doesn't exist in session_state
    if "conversation_memory" not in st.session_state:
        st.session_state.conversation_memory = ConversationBufferMemory()

    template = """
    System: Play the role of a friendly immigration lawyer. Respond to questions in detail, in the same language as the human's most recent question. If they ask a question in Spanish, you should answer in Spanish. If they ask a question in French, you should answer in French. And so on, for every language.
   
    {conversation_text}
    
    Human: {query}

    Lawyer: """

    prompt = template.format(query=query, conversation_text=st.session_state.conversation_memory.load_memory_variables({})['history'])

    docs = docsearch.similarity_search(query, include_metadata=True)

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    conversation = ConversationChain(
        llm=llm, verbose=True, memory=st.session_state.conversation_memory
    )

    with st.spinner('Processing your question...'):
        # Update conversation memory with user input
        st.session_state.conversation_memory.update_memory_variables({"input": query})

        # Generate prompt with updated conversation history
        prompt = template.format(query=query, conversation_text=st.session_state.conversation_memory.load_memory_variables({})['history'])

        result = conversation.predict(input=prompt)

    st.header("Prompt")
    st.write(prompt)  # Display the prompt value

    st.header("Answer")
    st.write(result)
    st.session_state.conversation_memory.save_context({"input": query}, {"output": result})

    # Display search results
    if docs:
        st.header("Search Results")
        st.write(f"Total search results: {len(docs)}")  # Display the number of results
        for index, doc in enumerate(docs, 1):
            st.write(f"Result {index}:")
            st.write(doc.page_content)  # Display each search result
            st.write("---")
    else:
        st.write("No results found.")



