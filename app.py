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

st.header("Immigration Q&A")

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
        background-color: #ddedee;
        color: #000;
    }
    .stApp h1 a, .stApp h2 a, .stApp h3 a, .stApp h4 a, .stApp h5 a, .stApp h6 a,
    .stApp h1 svg, .stApp h2 svg, .stApp h3 svg, .stApp h4 svg, .stApp h5 svg, .stApp h6 svg {
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

    template = """
    System: Play the role of a friendly immigration lawyer. Respond to questions in detail, in the same language as the human's most recent question. If they ask a question in Spanish, you should answer in Spanish. If they ask a question in French, you should answer in French. And so on, for every language.
   
    {conversation_text}  
    """

    # Retrieve the conversation history from the session state
    conversation_text = st.session_state.conversation.get_conversation()

    # Generate prompt with updated conversation history
    prompt = template.format(conversation_text=conversation_text)

    llm = OpenAI(temperature=0, openai_api_key=OPENAI_API_KEY, model_name="gpt-3.5-turbo")
    memory = ConversationBufferMemory()
    conversation = ConversationChain(llm=llm, verbose=True, memory=memory)
    from langchain.chains.question_answering import load_qa_chain
    chain = load_qa_chain(llm, chain_type="stuff")

    docs = docsearch.similarity_search(query, k=10)

    unique_ids = set()  # Set to track unique document IDs
    filtered_results = []

    for result in docs:
        document_id = result.metadata["id"]

        if document_id not in unique_ids:
            unique_ids.add(document_id)
            filtered_results.append(result)

    with st.spinner('Processing your question...'):
        result = chain.run(input_documents=filtered_results, question=prompt)


    # Add the AI's response to the conversation history
    st.session_state.conversation.add_message('AI', result)

    st.header("Prompt")
    st.write(prompt)  # Display the prompt value

    st.header("Answer")
    st.write(result)  # Display the AI-generated answer


    # Debug: Check the docs variable
    print(f"Number of docs: {len(docs)}")
    for i, doc in enumerate(docs):
        print(f"Doc {i}: {doc.page_content}")

    # Display search results
    if docs:
        st.header("Search Results")
        st.write(f"Total search results: {len(docs)}")  # Display the number of results

        for index, doc in enumerate(docs, 1):
            # Debug: Check each loop iteration
            print(f"Displaying result {index}")
            st.write(f"Result {index}:")
            st.write(doc.page_content)  # Display each search result
            st.write("---")
    else:
        st.write("No results found.")
        
    for i in range(min(3, len(docs))):
        st.write(docs[i].page_content[:500])
