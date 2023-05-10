import streamlit as st
from langchain.document_loaders import UnstructuredHTMLLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.vectorstores import Chroma, Pinecone
from langchain.embeddings.openai import OpenAIEmbeddings
import pinecone

st.set_page_config(page_title="Immigration Q&A", layout="wide", initial_sidebar_state="expanded")

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
col1, col2 = st.columns([1, 3])

# Display the image in the left column
image_path = 'statue.png'
col1.image(image_path, width=300)

# Display the text in the right column
col2.markdown("""
**Ask a question about immigration to the United States in any language, and our artificial intelligence will answer based on the USCIS handbook.**

*Legal Disclaimer: This platform is not a substitute for professional legal advice. The answers provided are based on the USCIS handbook and may not cover all aspects of your specific situation. For personalized guidance, please consult an immigration attorney.*

Ask a question about immigration to the United States in any language, and our artificial intelligence will answer based on the USCIS handbook.

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

if submit_button:
        template = """
        Lawyer: Hello! I am your friendly immigration lawyer. Any question you ask, I will answer in the same language as your question. How can I assist you today?

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
