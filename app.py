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
        background-color: #ADD8E6;
    }
    .anchor svg {
        display: none;
    }
</style>
"""
st.markdown(custom_css, unsafe_allow_html=True)



# Create two columns: one for the image and one for the text
col1, col2 = st.columns([1, 2])

# Display the image in the left column
image_path = 'statue.png'
col1.image(image_path, width=350)

# Display the text in the right column
col2.markdown("""
Ask a question about immigration to the United States in any language, and our artificial intelligence will answer based on the [USCIS policy manual](https://www.uscis.gov/policy-manual).

Haga una pregunta sobre inmigración a los Estados Unidos en cualquier idioma y nuestra inteligencia artificial responderá según el manual del USCIS.

用任何语言提问有关美国移民的问题，我们的人工智能将根据美国公民及移民服务局的手册进行回答。

Magtanong tungkol sa imigrasyon sa Estados Unidos sa anumang wika, at sasagutin ng aming artificial intelligence batay sa USCIS handbook.

Hỏi về nhập cư vào Hoa Kỳ bằng bất kỳ ngôn ngữ nào, và trí thông minh nhân tạo của chúng tôi sẽ trả lời dựa trên sách hướng dẫn của USCIS.

Báa ìbéèrè nípa ìjìmìn-ìlú sí Amẹ́ríkà ní èdè kankan, kí àṣà ìmọ̀ ọ̀rọ̀ wa ṣe àpèsè láti ìwé kikọ USCIS.

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

# Legal Disclaimer
col2.markdown("""
*Legal Disclaimer: This platform is meant for informational purposes only. It is not affiliated with USCIS or any other governmental organization, and is not a substitute for professional legal advice. The answers provided are based on the USCIS policy manual and may not cover all aspects of your specific situation. For personalized guidance, please consult an immigration attorney.*
""")


# Initialize the conversation memory
memory = ConversationBufferMemory()

if submit_button:
    template = """
    System: Play the role of a friendly immigration lawyer. Respond to questions in detail, in the same language as the human's most recent question. If they ask a question in Spanish, you should answer in Spanish. If they ask a question in French, you should answer in French. And so on, for every language.
   
    {conversation_text}
    
    Human: {query}

    Lawyer: """

    
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
        result = conversation.predict(input=prompt)

    st.header("Answer")
    st.write(result)
    st.session_state.conversation_memory.save_context({"input": query}, {"output": result})


